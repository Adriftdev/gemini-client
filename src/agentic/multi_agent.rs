use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    agentic::{
        build_system_instruction, build_user_content, extract_text_response,
        request_with_json_response,
    },
    types::GenerateContentResponse,
    GeminiError,
};

use super::{
    rag::{RagConfig, RagError, RagSession, Retriever},
    tool_runtime::{execute_tool_loop, AgentTools, ModelBackend, ToolRuntimeConfig},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentSpec {
    pub role: String,
    pub model: Option<String>,
    pub instruction: String,
}

impl AgentSpec {
    pub fn new(role: &str, instruction: &str) -> Self {
        Self {
            role: role.to_string(),
            model: None,
            instruction: instruction.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Assignment {
    pub agent_role: String,
    pub task: String,
    pub success_criteria: String,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    pub needs_rag: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    pub agent_role: String,
    pub assignment_task: String,
    pub content: String,
    pub tool_trace: Option<super::tool_runtime::ToolTrace>,
    #[serde(default)]
    pub citations: Vec<String>,
    pub revision_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReviewRecord {
    pub agent_role: String,
    pub assignment_task: String,
    pub decision: String,
    pub feedback: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BlackboardEntry {
    Assignment(Assignment),
    Artifact(Artifact),
    Review(ReviewRecord),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Blackboard {
    pub entries: Vec<BlackboardEntry>,
}

impl Blackboard {
    pub fn push_assignment(&mut self, assignment: Assignment) {
        self.entries.push(BlackboardEntry::Assignment(assignment));
    }

    pub fn push_artifact(&mut self, artifact: Artifact) {
        self.entries.push(BlackboardEntry::Artifact(artifact));
    }

    pub fn push_review(&mut self, review: ReviewRecord) {
        self.entries.push(BlackboardEntry::Review(review));
    }
}

#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    pub max_assignments: usize,
    pub max_revisions: usize,
    pub tool_runtime: ToolRuntimeConfig,
    pub rag: RagConfig,
    pub agents: Vec<AgentSpec>,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            max_assignments: 3,
            max_revisions: 1,
            tool_runtime: ToolRuntimeConfig::default(),
            rag: RagConfig::default(),
            agents: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct SupervisorOutcome {
    pub final_answer: String,
    pub blackboard: Blackboard,
    pub assignments: Vec<Assignment>,
    pub accepted_artifacts: Vec<Artifact>,
    pub raw_response: GenerateContentResponse,
}

#[derive(Debug, thiserror::Error)]
pub enum OrchestrationError {
    #[error(transparent)]
    Backend(#[from] GeminiError),
    #[error(transparent)]
    Rag(#[from] RagError),
    #[error("Failed to parse JSON for {context}: {error} (payload: {data})")]
    Json {
        context: String,
        data: String,
        #[source]
        error: serde_json::Error,
    },
    #[error("The model did not return any text for {context}")]
    NoTextResponse { context: String },
    #[error("The supervisor produced no assignments")]
    NoAssignments,
    #[error("The supervisor produced {count} assignments, which exceeds the limit of {max}")]
    AssignmentLimitExceeded { count: usize, max: usize },
    #[error("Assignment referenced an unknown allowed tool: {tool_name}")]
    UnknownAllowedTool { tool_name: String },
    #[error("RAG was required for assignment {task:?}, but no retriever was provided")]
    RagRequiredButUnavailable { task: String },
    #[error("No artifacts were accepted by the reviewer")]
    NoAcceptedArtifacts,
}

pub struct SupervisorWorkflow<'a, B> {
    backend: &'a B,
    config: SupervisorConfig,
}

impl<'a, B> SupervisorWorkflow<'a, B>
where
    B: ModelBackend,
{
    pub fn new(backend: &'a B, config: SupervisorConfig) -> Self {
        Self { backend, config }
    }

    pub async fn run<R: Retriever>(
        &self,
        model: &str,
        task: &str,
        tools: Option<&AgentTools>,
        rag: Option<&R>,
    ) -> Result<SupervisorOutcome, OrchestrationError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.supervisor.run",
            model = model,
            max_assignments = self.config.max_assignments,
            max_revisions = self.config.max_revisions,
            has_tools = tools.is_some(),
            has_rag = rag.is_some()
        );
        crate::telemetry::telemetry_info!("supervisor.run started");
        let assignments = self.supervisor_assignments(model, task, tools).await?;
        crate::telemetry::telemetry_info!(
            assignments_count = assignments.len(),
            "supervisor produced assignments"
        );
        let mut blackboard = Blackboard::default();
        let mut accepted_artifacts = vec![];

        for assignment in &assignments {
            let _assignment_span = crate::telemetry::telemetry_span_guard!(
                debug,
                "gemini_client_rs.supervisor.assignment",
                agent_role = assignment.agent_role.as_str(),
                allowed_tools_count = assignment.allowed_tools.len(),
                needs_rag = assignment.needs_rag,
                revision_count = 0usize
            );
            blackboard.push_assignment(assignment.clone());

            let mut artifact = self
                .execute_assignment(model, task, assignment, tools, rag, 0)
                .await?;
            blackboard.push_artifact(artifact.clone());

            let mut review = self
                .review_artifact(model, task, assignment, &artifact, &blackboard)
                .await?;
            blackboard.push_review(review.clone());
            crate::telemetry::telemetry_debug!(
                agent_role = assignment.agent_role.as_str(),
                decision = review.decision.as_str(),
                "supervisor.review completed"
            );

            if review.decision == "revise" && self.config.max_revisions > 0 {
                crate::telemetry::telemetry_warn!(
                    agent_role = assignment.agent_role.as_str(),
                    "supervisor assignment requested revision"
                );
                artifact = self
                    .execute_assignment(model, task, assignment, tools, rag, 1)
                    .await?;
                artifact.content = format!(
                    "{}\n\nReviewer feedback addressed: {}",
                    artifact.content, review.feedback
                );
                blackboard.push_artifact(artifact.clone());
                review = self
                    .review_artifact(model, task, assignment, &artifact, &blackboard)
                    .await?;
                blackboard.push_review(review.clone());
                crate::telemetry::telemetry_debug!(
                    agent_role = assignment.agent_role.as_str(),
                    decision = review.decision.as_str(),
                    revision_count = artifact.revision_count,
                    "supervisor.review completed after revision"
                );
            }

            if review.decision == "accept" {
                accepted_artifacts.push(artifact);
            }
        }

        if accepted_artifacts.is_empty() {
            crate::telemetry::telemetry_warn!("supervisor.run finished with no accepted artifacts");
            return Err(OrchestrationError::NoAcceptedArtifacts);
        }

        let raw_response = self
            .synthesize(model, task, &accepted_artifacts, &blackboard)
            .await?;
        let final_answer = extract_text_response(&raw_response).ok_or_else(|| {
            OrchestrationError::NoTextResponse {
                context: "synthesizer".to_string(),
            }
        })?;
        crate::telemetry::telemetry_info!(
            assignments_count = assignments.len(),
            accepted_artifacts_count = accepted_artifacts.len(),
            "supervisor.run completed"
        );

        Ok(SupervisorOutcome {
            final_answer,
            blackboard,
            assignments,
            accepted_artifacts,
            raw_response,
        })
    }

    async fn supervisor_assignments(
        &self,
        model: &str,
        task: &str,
        tools: Option<&AgentTools>,
    ) -> Result<Vec<Assignment>, OrchestrationError> {
        let prompt = format!(
            "Create a JSON assignment list for the task.\nReturn at most {} assignments.\nEach assignment must contain agent_role, task, success_criteria, allowed_tools, and needs_rag.\nUse only these tool names when needed: {:?}.\nDefault the agent role to \"worker\" unless there is a strong reason to use another role.\nTask: {}",
            self.config.max_assignments,
            tools
                .map(|available| available.available_tool_names())
                .unwrap_or_default(),
            task
        );
        let schema = json!({
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_role": { "type": "string" },
                            "task": { "type": "string" },
                            "success_criteria": { "type": "string" },
                            "allowed_tools": {
                                "type": "array",
                                "items": { "type": "string" }
                            },
                            "needs_rag": { "type": "boolean" }
                        },
                        "required": [
                            "agent_role",
                            "task",
                            "success_criteria",
                            "allowed_tools",
                            "needs_rag"
                        ]
                    }
                }
            },
            "required": ["assignments"]
        });
        let request =
            request_with_json_response(Some(self.role_instruction("supervisor")), prompt, schema);
        let response = self.backend.generate_content(model, &request).await?;
        let payload_text =
            extract_text_response(&response).ok_or_else(|| OrchestrationError::NoTextResponse {
                context: "supervisor".to_string(),
            })?;
        let payload =
            serde_json::from_str::<AssignmentEnvelope>(&payload_text).map_err(|error| {
                OrchestrationError::Json {
                    context: "supervisor".to_string(),
                    data: payload_text.clone(),
                    error,
                }
            })?;

        if payload.assignments.is_empty() {
            crate::telemetry::telemetry_warn!("supervisor produced no assignments");
            return Err(OrchestrationError::NoAssignments);
        }

        if payload.assignments.len() > self.config.max_assignments {
            crate::telemetry::telemetry_warn!(
                assignments_count = payload.assignments.len(),
                max_assignments = self.config.max_assignments,
                "supervisor exceeded assignment limit"
            );
            return Err(OrchestrationError::AssignmentLimitExceeded {
                count: payload.assignments.len(),
                max: self.config.max_assignments,
            });
        }

        Ok(payload.assignments)
    }

    async fn execute_assignment<R: Retriever>(
        &self,
        model: &str,
        overall_task: &str,
        assignment: &Assignment,
        tools: Option<&AgentTools>,
        rag: Option<&R>,
        revision_count: usize,
    ) -> Result<Artifact, OrchestrationError> {
        let worker_model = self.role_model("worker", model);
        crate::telemetry::telemetry_debug!(
            agent_role = assignment.agent_role.as_str(),
            revision_count,
            "supervisor.assignment execution started"
        );

        if assignment.needs_rag {
            let Some(retriever) = rag else {
                crate::telemetry::telemetry_warn!(
                    agent_role = assignment.agent_role.as_str(),
                    "supervisor assignment required rag but no retriever was provided"
                );
                return Err(OrchestrationError::RagRequiredButUnavailable {
                    task: assignment.task.clone(),
                });
            };
            let session = RagSession::new(self.backend, retriever, self.config.rag.clone());
            let response = session
                .answer(
                    worker_model,
                    &format!(
                        "Overall task: {overall_task}\nAssignment: {}\nSuccess criteria: {}",
                        assignment.task, assignment.success_criteria
                    ),
                    Some(self.role_instruction("worker")),
                )
                .await?;
            crate::telemetry::telemetry_debug!(
                agent_role = assignment.agent_role.as_str(),
                citation_count = response.cited_chunk_ids.len(),
                revision_count,
                "supervisor.assignment completed via rag"
            );
            return Ok(Artifact {
                agent_role: assignment.agent_role.clone(),
                assignment_task: assignment.task.clone(),
                content: response.answer,
                tool_trace: None,
                citations: response.cited_chunk_ids,
                revision_count,
            });
        }

        if let Some(agent_tools) = tools {
            let missing = agent_tools.missing_allowed_names(&assignment.allowed_tools);
            if let Some(tool_name) = missing.first() {
                crate::telemetry::telemetry_warn!(
                    agent_role = assignment.agent_role.as_str(),
                    tool_name = tool_name.as_str(),
                    "supervisor assignment referenced an unknown allowed tool"
                );
                return Err(OrchestrationError::UnknownAllowedTool {
                    tool_name: tool_name.clone(),
                });
            }

            if !assignment.allowed_tools.is_empty() {
                let selection = agent_tools.select(&assignment.allowed_tools);
                let request = crate::types::GenerateContentRequest {
                    system_instruction: build_system_instruction(Some(
                        self.role_instruction("worker"),
                    )),
                    contents: vec![build_user_content(&format!(
                        "Overall task: {overall_task}\nAssignment: {}\nSuccess criteria: {}",
                        assignment.task, assignment.success_criteria
                    ))],
                    tools: selection.tools().to_vec(),
                    tool_config: None,
                    generation_config: None,
                };
                let result = execute_tool_loop(
                    self.backend,
                    worker_model,
                    request,
                    Some(&selection),
                    &self.config.tool_runtime,
                )
                .await?;
                crate::telemetry::telemetry_debug!(
                    agent_role = assignment.agent_role.as_str(),
                    tool_call_count = result.trace.calls.len(),
                    round_trips = result.trace.round_trips,
                    revision_count,
                    "supervisor.assignment completed via tool runtime"
                );
                return Ok(Artifact {
                    agent_role: assignment.agent_role.clone(),
                    assignment_task: assignment.task.clone(),
                    content: extract_text_response(&result.response).unwrap_or_else(|| {
                        serde_json::to_string(&result.response).unwrap_or_default()
                    }),
                    tool_trace: Some(result.trace),
                    citations: vec![],
                    revision_count,
                });
            }
        }

        let request = crate::types::GenerateContentRequest {
            system_instruction: build_system_instruction(Some(self.role_instruction("worker"))),
            contents: vec![build_user_content(&format!(
                "Overall task: {overall_task}\nAssignment: {}\nSuccess criteria: {}",
                assignment.task, assignment.success_criteria
            ))],
            tools: vec![],
            tool_config: None,
            generation_config: None,
        };
        let response = self
            .backend
            .generate_content(worker_model, &request)
            .await?;
        let content =
            extract_text_response(&response).ok_or_else(|| OrchestrationError::NoTextResponse {
                context: "worker".to_string(),
            })?;
        crate::telemetry::telemetry_debug!(
            agent_role = assignment.agent_role.as_str(),
            revision_count,
            "supervisor.assignment completed via plain generation"
        );

        Ok(Artifact {
            agent_role: assignment.agent_role.clone(),
            assignment_task: assignment.task.clone(),
            content,
            tool_trace: None,
            citations: vec![],
            revision_count,
        })
    }

    async fn review_artifact(
        &self,
        model: &str,
        overall_task: &str,
        assignment: &Assignment,
        artifact: &Artifact,
        blackboard: &Blackboard,
    ) -> Result<ReviewRecord, OrchestrationError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            debug,
            "gemini_client_rs.supervisor.review",
            agent_role = assignment.agent_role.as_str(),
            revision_count = artifact.revision_count
        );
        let request = request_with_json_response(
            Some(self.role_instruction("reviewer")),
            format!(
                "Review the worker artifact.\nReturn JSON with decision and feedback. Decision must be accept or revise.\nOverall task: {overall_task}\nAssignment: {}\nSuccess criteria: {}\nArtifact: {}\nBlackboard entries: {}",
                assignment.task,
                assignment.success_criteria,
                artifact.content,
                serde_json::to_string_pretty(&blackboard.entries).unwrap_or_else(|_| "[]".to_string())
            ),
            json!({
                "type": "object",
                "properties": {
                    "decision": { "type": "string" },
                    "feedback": { "type": "string" }
                },
                "required": ["decision", "feedback"]
            }),
        );
        let response = self
            .backend
            .generate_content(self.role_model("reviewer", model), &request)
            .await?;
        let payload_text =
            extract_text_response(&response).ok_or_else(|| OrchestrationError::NoTextResponse {
                context: "reviewer".to_string(),
            })?;
        let payload = serde_json::from_str::<ReviewPayload>(&payload_text).map_err(|error| {
            OrchestrationError::Json {
                context: "reviewer".to_string(),
                data: payload_text.clone(),
                error,
            }
        })?;
        crate::telemetry::telemetry_debug!(
            agent_role = assignment.agent_role.as_str(),
            decision = payload.decision.as_str(),
            revision_count = artifact.revision_count,
            "supervisor.review parsed reviewer decision"
        );

        Ok(ReviewRecord {
            agent_role: assignment.agent_role.clone(),
            assignment_task: assignment.task.clone(),
            decision: payload.decision,
            feedback: payload.feedback,
        })
    }

    async fn synthesize(
        &self,
        model: &str,
        task: &str,
        accepted_artifacts: &[Artifact],
        blackboard: &Blackboard,
    ) -> Result<GenerateContentResponse, OrchestrationError> {
        let request = crate::types::GenerateContentRequest {
            system_instruction: build_system_instruction(Some(
                self.role_instruction("synthesizer"),
            )),
            contents: vec![build_user_content(&format!(
                "Task: {task}\nAccepted artifacts: {}\nBlackboard: {}",
                serde_json::to_string_pretty(accepted_artifacts)
                    .unwrap_or_else(|_| "[]".to_string()),
                serde_json::to_string_pretty(&blackboard.entries)
                    .unwrap_or_else(|_| "[]".to_string())
            ))],
            tools: vec![],
            tool_config: None,
            generation_config: None,
        };

        let response = self
            .backend
            .generate_content(self.role_model("synthesizer", model), &request)
            .await
            .map_err(OrchestrationError::from)?;
        crate::telemetry::telemetry_debug!(
            accepted_artifacts_count = accepted_artifacts.len(),
            "supervisor.synthesize completed"
        );
        Ok(response)
    }

    fn role_instruction(&self, role: &str) -> &str {
        self.config
            .agents
            .iter()
            .find(|agent| agent.role == role)
            .map(|agent| agent.instruction.as_str())
            .unwrap_or(match role {
                "supervisor" => {
                    "You are the supervisor. Break the task into up to three concrete assignments."
                }
                "reviewer" => {
                    "You are the reviewer. Accept only artifacts that satisfy the assignment."
                }
                "synthesizer" => {
                    "You are the synthesizer. Combine accepted artifacts into the final answer."
                }
                _ => "You are the worker. Complete the assignment directly and clearly.",
            })
    }

    fn role_model<'b>(&'b self, role: &str, default_model: &'b str) -> &'b str {
        self.config
            .agents
            .iter()
            .find(|agent| agent.role == role)
            .and_then(|agent| agent.model.as_deref())
            .unwrap_or(default_model)
    }
}

#[derive(Debug, Deserialize)]
struct AssignmentEnvelope {
    assignments: Vec<Assignment>,
}

#[derive(Debug, Deserialize)]
struct ReviewPayload {
    decision: String,
    feedback: String,
}

#[cfg(test)]
mod tests {
    use super::{OrchestrationError, SupervisorConfig, SupervisorWorkflow};
    use crate::agentic::test_support::{response_with_text, ScriptedBackend, StaticRetriever};

    #[tokio::test]
    async fn supervisor_caps_assignments() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_text(
                r#"{"assignments":[
                    {"agent_role":"worker","task":"one","success_criteria":"one","allowed_tools":[],"needs_rag":false},
                    {"agent_role":"worker","task":"two","success_criteria":"two","allowed_tools":[],"needs_rag":false},
                    {"agent_role":"worker","task":"three","success_criteria":"three","allowed_tools":[],"needs_rag":false},
                    {"agent_role":"worker","task":"four","success_criteria":"four","allowed_tools":[],"needs_rag":false}
                ]}"#,
            ))
        })]);
        let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());
        let error = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect_err("assignment cap should be enforced");

        assert!(matches!(
            error,
            OrchestrationError::AssignmentLimitExceeded { count: 4, max: 3 }
        ));
    }

    #[tokio::test]
    async fn supervisor_stores_artifacts_and_reviews() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"assignments":[{"agent_role":"worker","task":"one","success_criteria":"done","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"accept","feedback":"good"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("final"))),
        ]);
        let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());
        let outcome = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("workflow should succeed");

        assert_eq!(outcome.accepted_artifacts.len(), 1);
        assert_eq!(outcome.blackboard.entries.len(), 3);
    }

    #[tokio::test]
    async fn supervisor_allows_a_single_revision_cycle() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"assignments":[{"agent_role":"worker","task":"one","success_criteria":"done","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"revise","feedback":"tighten it"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one revised"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"accept","feedback":"good"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("final"))),
        ]);
        let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());
        let outcome = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("workflow should succeed");

        assert_eq!(outcome.accepted_artifacts[0].revision_count, 1);
        assert!(outcome.final_answer.contains("final"));
    }

    #[tokio::test]
    async fn supervisor_requires_accepted_artifacts_for_synthesis() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"assignments":[{"agent_role":"worker","task":"one","success_criteria":"done","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"revise","feedback":"tighten it"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one revised"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"revise","feedback":"still not right"}"#,
                ))
            }),
        ]);
        let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());
        let error = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect_err("workflow should fail without accepted artifacts");

        assert!(matches!(error, OrchestrationError::NoAcceptedArtifacts));
    }
}

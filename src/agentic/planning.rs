use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agentic::{
        build_system_instruction, build_user_content, extract_text_response,
        request_with_json_response,
    },
    types::{GenerateContentRequest, GenerateContentResponse},
    GeminiError,
};

use super::{
    rag::{RagConfig, RagError, RagResponse, RagSession, Retriever},
    tool_runtime::{execute_tool_loop, AgentTools, ModelBackend, ToolRuntimeConfig},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanStep {
    pub id: String,
    pub title: String,
    pub instruction: String,
    pub success_criteria: String,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    pub needs_rag: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepResult {
    pub step_id: String,
    pub summary: String,
    pub raw_text: String,
    pub tool_trace: Option<super::tool_runtime::ToolTrace>,
    #[serde(default)]
    pub citations: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WorkingMemory {
    pub entries: HashMap<String, Value>,
}

impl WorkingMemory {
    pub fn insert(&mut self, step_id: &str, result: &StepResult) -> Result<(), PlanningError> {
        self.entries.insert(
            step_id.to_string(),
            serde_json::to_value(result).map_err(|error| PlanningError::MemorySerialization {
                step_id: step_id.to_string(),
                error,
            })?,
        );
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PlanningConfig {
    pub max_steps: usize,
    pub max_step_retries: usize,
    pub max_replans: usize,
    pub tool_runtime: ToolRuntimeConfig,
    pub rag: RagConfig,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            max_steps: 8,
            max_step_retries: 1,
            max_replans: 1,
            tool_runtime: ToolRuntimeConfig::default(),
            rag: RagConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanningEvaluationRecord {
    pub step_id: String,
    pub decision: String,
    pub feedback: String,
}

#[derive(Debug, Clone)]
pub struct PlanningTrace {
    pub initial_plan: Plan,
    pub replans: Vec<Plan>,
    pub step_results: Vec<StepResult>,
    pub evaluations: Vec<PlanningEvaluationRecord>,
}

#[derive(Debug, Clone)]
pub struct PlanOutcome {
    pub final_answer: String,
    pub trace: PlanningTrace,
    pub working_memory: WorkingMemory,
    pub raw_response: GenerateContentResponse,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanningError {
    #[error(transparent)]
    Backend(Box<GeminiError>),
    #[error(transparent)]
    Rag(Box<RagError>),
    #[error("The planner returned no steps")]
    EmptyPlan,
    #[error("The planner returned {count} steps, which exceeds the limit of {max}")]
    PlanTooLarge { count: usize, max: usize },
    #[error("Failed to parse JSON for {context}: {error} (payload: {data})")]
    Json {
        context: String,
        data: String,
        #[source]
        error: serde_json::Error,
    },
    #[error("The model did not return any text for {context}")]
    NoTextResponse { context: String },
    #[error("RAG was required for step {step_id}, but no retriever was provided")]
    RagRequiredButUnavailable { step_id: String },
    #[error("Step {step_id} referenced an unknown allowed tool: {tool_name}")]
    UnknownAllowedTool { step_id: String, tool_name: String },
    #[error("Retry limit exceeded for step {step_id}")]
    RetryLimitExceeded { step_id: String },
    #[error("Replan limit exceeded after {max_replans} replans")]
    ReplanLimitExceeded { max_replans: usize },
    #[error("Execution failed for step {step_id}: {feedback}")]
    ExecutionFailed { step_id: String, feedback: String },
    #[error("Failed to serialize working-memory entry for {step_id}: {error}")]
    MemorySerialization {
        step_id: String,
        #[source]
        error: serde_json::Error,
    },
}

pub struct PlanningSession<'a, B> {
    backend: &'a B,
    config: PlanningConfig,
}

impl<'a, B> PlanningSession<'a, B>
where
    B: ModelBackend,
{
    pub fn new(backend: &'a B, config: PlanningConfig) -> Self {
        Self { backend, config }
    }

    pub async fn run<R: Retriever>(
        &self,
        model: &str,
        task: &str,
        tools: Option<&AgentTools>,
        rag: Option<&R>,
    ) -> Result<PlanOutcome, PlanningError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.plan.run",
            model = model,
            max_steps = self.config.max_steps,
            max_step_retries = self.config.max_step_retries,
            max_replans = self.config.max_replans,
            has_tools = tools.is_some(),
            has_rag = rag.is_some()
        );
        crate::telemetry::telemetry_info!("plan.run started");
        let mut working_memory = WorkingMemory::default();
        let mut trace = PlanningTrace {
            initial_plan: self.plan(model, task, tools, &working_memory, None).await?,
            replans: vec![],
            step_results: vec![],
            evaluations: vec![],
        };
        crate::telemetry::telemetry_info!(
            step_count = trace.initial_plan.steps.len(),
            "plan.run planner produced initial plan"
        );

        let mut current_plan = trace.initial_plan.clone();
        let mut step_retries = HashMap::<String, usize>::new();
        let mut replans = 0usize;
        let mut step_index = 0usize;

        while step_index < current_plan.steps.len() {
            let step = current_plan.steps[step_index].clone();
            let _step_span = crate::telemetry::telemetry_span_guard!(
                debug,
                "gemini_client_rs.plan.step",
                step_id = step.id.as_str(),
                allowed_tools_count = step.allowed_tools.len(),
                needs_rag = step.needs_rag
            );
            crate::telemetry::telemetry_debug!("plan.step started");
            let step_result = self
                .execute_step(model, task, &step, tools, rag, &working_memory)
                .await?;
            working_memory.insert(&step.id, &step_result)?;
            trace.step_results.push(step_result.clone());

            let _evaluate_span = crate::telemetry::telemetry_span_guard!(
                debug,
                "gemini_client_rs.plan.evaluate",
                step_id = step.id.as_str()
            );
            let evaluation = self
                .evaluate_step(model, task, &step, &step_result, &working_memory)
                .await?;
            trace.evaluations.push(evaluation.clone());
            crate::telemetry::telemetry_debug!(
                step_id = step.id.as_str(),
                decision = evaluation.decision.as_str(),
                "plan.evaluate completed"
            );

            match EvaluationDecision::from_label(&evaluation.decision) {
                EvaluationDecision::Pass => {
                    step_index += 1;
                }
                EvaluationDecision::RetryStep => {
                    crate::telemetry::telemetry_warn!(
                        step_id = step.id.as_str(),
                        decision = evaluation.decision.as_str(),
                        "plan.step requested retry"
                    );
                    let retries = step_retries.entry(step.id.clone()).or_insert(0);
                    if *retries >= self.config.max_step_retries {
                        crate::telemetry::telemetry_warn!(
                            step_id = step.id.as_str(),
                            max_step_retries = self.config.max_step_retries,
                            "plan.step exceeded retry limit"
                        );
                        return Err(PlanningError::RetryLimitExceeded {
                            step_id: step.id.clone(),
                        });
                    }
                    *retries += 1;
                }
                EvaluationDecision::Replan => {
                    crate::telemetry::telemetry_warn!(
                        step_id = step.id.as_str(),
                        "plan.run requested replan"
                    );
                    if replans >= self.config.max_replans {
                        crate::telemetry::telemetry_warn!(
                            max_replans = self.config.max_replans,
                            "plan.run exceeded replan limit"
                        );
                        return Err(PlanningError::ReplanLimitExceeded {
                            max_replans: self.config.max_replans,
                        });
                    }
                    replans += 1;
                    current_plan = self
                        .plan(
                            model,
                            task,
                            tools,
                            &working_memory,
                            Some(&evaluation.feedback),
                        )
                        .await?;
                    trace.replans.push(current_plan.clone());
                    crate::telemetry::telemetry_info!(
                        replan_count = trace.replans.len(),
                        step_count = current_plan.steps.len(),
                        "plan.run planner produced revised plan"
                    );
                    step_retries.clear();
                    step_index = 0;
                }
                EvaluationDecision::Fail => {
                    crate::telemetry::telemetry_warn!(
                        step_id = step.id.as_str(),
                        "plan.step failed evaluation"
                    );
                    return Err(PlanningError::ExecutionFailed {
                        step_id: step.id,
                        feedback: evaluation.feedback,
                    });
                }
            }
        }

        let raw_response = self.final_synthesis(model, task, &working_memory).await?;
        let final_answer =
            extract_text_response(&raw_response).ok_or_else(|| PlanningError::NoTextResponse {
                context: "final synthesis".to_string(),
            })?;
        crate::telemetry::telemetry_info!(
            executed_steps = trace.step_results.len(),
            "plan.run completed"
        );

        Ok(PlanOutcome {
            final_answer,
            trace,
            working_memory,
            raw_response,
        })
    }

    async fn plan(
        &self,
        model: &str,
        task: &str,
        tools: Option<&AgentTools>,
        working_memory: &WorkingMemory,
        replanning_feedback: Option<&str>,
    ) -> Result<Plan, PlanningError> {
        let tool_names = tools
            .map(|available| available.available_tool_names())
            .unwrap_or_default();
        let prompt = format!(
            "Create a JSON plan for the task.\nReturn at most {} steps.\nEach step must contain id, title, instruction, success_criteria, allowed_tools, and needs_rag.\nUse only these available tools: {:?}.\nSet needs_rag to true only when the step requires retrieved knowledge from an external corpus.\nCurrent working memory: {}\nTask: {}\n{}",
            self.config.max_steps,
            tool_names,
            serde_json::to_string_pretty(&working_memory.entries).unwrap_or_else(|_| "{}".to_string()),
            task,
            replanning_feedback
                .map(|feedback| format!("Replanning feedback: {feedback}"))
                .unwrap_or_default()
        );
        let schema = json!({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string" },
                            "title": { "type": "string" },
                            "instruction": { "type": "string" },
                            "success_criteria": { "type": "string" },
                            "allowed_tools": {
                                "type": "array",
                                "items": { "type": "string" }
                            },
                            "needs_rag": { "type": "boolean" }
                        },
                        "required": [
                            "id",
                            "title",
                            "instruction",
                            "success_criteria",
                            "allowed_tools",
                            "needs_rag"
                        ]
                    }
                }
            },
            "required": ["steps"]
        });

        let request = request_with_json_response(
            Some("You are a planning engine. Produce concise, executable plans."),
            prompt,
            schema,
        );
        let response = self.backend.generate_content(model, &request).await?;
        let payload_text =
            extract_text_response(&response).ok_or_else(|| PlanningError::NoTextResponse {
                context: "planner".to_string(),
            })?;
        let plan =
            serde_json::from_str::<Plan>(&payload_text).map_err(|error| PlanningError::Json {
                context: "planner".to_string(),
                data: payload_text.clone(),
                error,
            })?;

        if plan.steps.is_empty() {
            crate::telemetry::telemetry_warn!("plan planner returned an empty plan");
            return Err(PlanningError::EmptyPlan);
        }

        if plan.steps.len() > self.config.max_steps {
            crate::telemetry::telemetry_warn!(
                step_count = plan.steps.len(),
                max_steps = self.config.max_steps,
                "plan planner exceeded max steps"
            );
            return Err(PlanningError::PlanTooLarge {
                count: plan.steps.len(),
                max: self.config.max_steps,
            });
        }

        Ok(plan)
    }

    async fn execute_step<R: Retriever>(
        &self,
        model: &str,
        task: &str,
        step: &PlanStep,
        tools: Option<&AgentTools>,
        rag: Option<&R>,
        working_memory: &WorkingMemory,
    ) -> Result<StepResult, PlanningError> {
        if step.needs_rag {
            let Some(retriever) = rag else {
                crate::telemetry::telemetry_warn!(
                    step_id = step.id.as_str(),
                    "plan.step required rag but no retriever was provided"
                );
                return Err(PlanningError::RagRequiredButUnavailable {
                    step_id: step.id.clone(),
                });
            };
            let session = RagSession::new(self.backend, retriever, self.config.rag.clone());
            let rag_response = session
                .answer(
                    model,
                    &format!(
                        "Task: {task}\nStep: {}\nInstruction: {}\nWorking memory: {}",
                        step.title,
                        step.instruction,
                        serde_json::to_string_pretty(&working_memory.entries)
                            .unwrap_or_else(|_| "{}".to_string())
                    ),
                    Some("You are executing a single plan step with retrieval support."),
                )
                .await?;
            crate::telemetry::telemetry_debug!(
                step_id = step.id.as_str(),
                citation_count = rag_response.cited_chunk_ids.len(),
                "plan.step completed via rag"
            );
            return Ok(step_result_from_rag(step, rag_response));
        }

        if let Some(agent_tools) = tools {
            let missing = agent_tools.missing_allowed_names(&step.allowed_tools);
            if let Some(tool_name) = missing.first() {
                crate::telemetry::telemetry_warn!(
                    step_id = step.id.as_str(),
                    tool_name = tool_name.as_str(),
                    "plan.step referenced an unknown allowed tool"
                );
                return Err(PlanningError::UnknownAllowedTool {
                    step_id: step.id.clone(),
                    tool_name: tool_name.clone(),
                });
            }

            if !step.allowed_tools.is_empty() {
                let selection = agent_tools.select(&step.allowed_tools);
                let request = GenerateContentRequest {
                    system_instruction: build_system_instruction(Some(
                        "You are executing a single plan step. Use tools when they help satisfy the step criteria.",
                    )),
                    contents: vec![build_user_content(&format!(
                        "Overall task: {task}\nStep title: {}\nInstruction: {}\nSuccess criteria: {}\nWorking memory: {}",
                        step.title,
                        step.instruction,
                        step.success_criteria,
                        serde_json::to_string_pretty(&working_memory.entries)
                            .unwrap_or_else(|_| "{}".to_string())
                    ))],
                    tools: selection.tools().to_vec(),
                    tool_config: None,
                    generation_config: None,
                };
                let tool_result = execute_tool_loop(
                    self.backend,
                    model,
                    request,
                    Some(&selection),
                    &self.config.tool_runtime,
                )
                .await?;
                let raw_text = response_text_or_json(&tool_result.response);
                crate::telemetry::telemetry_debug!(
                    step_id = step.id.as_str(),
                    tool_call_count = tool_result.trace.calls.len(),
                    round_trips = tool_result.trace.round_trips,
                    "plan.step completed via tool runtime"
                );
                return Ok(StepResult {
                    step_id: step.id.clone(),
                    summary: summarize(&raw_text),
                    raw_text,
                    tool_trace: Some(tool_result.trace),
                    citations: vec![],
                });
            }
        }

        let request = GenerateContentRequest {
            system_instruction: build_system_instruction(Some(
                "You are executing a single plan step. Respond with the result only.",
            )),
            contents: vec![build_user_content(&format!(
                "Overall task: {task}\nStep title: {}\nInstruction: {}\nSuccess criteria: {}\nWorking memory: {}",
                step.title,
                step.instruction,
                step.success_criteria,
                serde_json::to_string_pretty(&working_memory.entries)
                    .unwrap_or_else(|_| "{}".to_string())
            ))],
            tools: vec![],
            tool_config: None,
            generation_config: None,
        };
        let response = self.backend.generate_content(model, &request).await?;
        let raw_text = response_text_or_json(&response);
        crate::telemetry::telemetry_debug!(
            step_id = step.id.as_str(),
            "plan.step completed via plain generation"
        );

        Ok(StepResult {
            step_id: step.id.clone(),
            summary: summarize(&raw_text),
            raw_text,
            tool_trace: None,
            citations: vec![],
        })
    }

    async fn evaluate_step(
        &self,
        model: &str,
        task: &str,
        step: &PlanStep,
        step_result: &StepResult,
        working_memory: &WorkingMemory,
    ) -> Result<PlanningEvaluationRecord, PlanningError> {
        let prompt = format!(
            "Evaluate whether the step result satisfies the success criteria.\nReturn JSON with keys decision and feedback.\nValid decisions are pass, retry_step, replan, fail.\n\nTask: {task}\nStep: {}\nSuccess criteria: {}\nStep result: {}\nWorking memory: {}",
            step.title,
            step.success_criteria,
            serde_json::to_string_pretty(step_result).unwrap_or_else(|_| "{}".to_string()),
            serde_json::to_string_pretty(&working_memory.entries).unwrap_or_else(|_| "{}".to_string())
        );
        let schema = json!({
            "type": "object",
            "properties": {
                "decision": { "type": "string" },
                "feedback": { "type": "string" }
            },
            "required": ["decision", "feedback"]
        });
        let request = request_with_json_response(
            Some("You are a strict evaluator. Prefer pass only when the step clearly satisfies the criteria."),
            prompt,
            schema,
        );
        let response = self.backend.generate_content(model, &request).await?;
        let payload_text =
            extract_text_response(&response).ok_or_else(|| PlanningError::NoTextResponse {
                context: "evaluator".to_string(),
            })?;
        let payload =
            serde_json::from_str::<EvaluationPayload>(&payload_text).map_err(|error| {
                PlanningError::Json {
                    context: "evaluator".to_string(),
                    data: payload_text.clone(),
                    error,
                }
            })?;

        Ok(PlanningEvaluationRecord {
            step_id: step.id.clone(),
            decision: payload.decision,
            feedback: payload.feedback,
        })
    }

    async fn final_synthesis(
        &self,
        model: &str,
        task: &str,
        working_memory: &WorkingMemory,
    ) -> Result<GenerateContentResponse, PlanningError> {
        let request = GenerateContentRequest {
            system_instruction: build_system_instruction(Some(
                "You are the final synthesizer. Produce a concise final answer from the working memory.",
            )),
            contents: vec![build_user_content(&format!(
                "Task: {task}\nWorking memory: {}",
                serde_json::to_string_pretty(&working_memory.entries)
                    .unwrap_or_else(|_| "{}".to_string())
            ))],
            tools: vec![],
            tool_config: None,
            generation_config: None,
        };

        let response = self
            .backend
            .generate_content(model, &request)
            .await
            .map_err(PlanningError::from)?;
        crate::telemetry::telemetry_debug!("plan.final_synthesis completed");
        Ok(response)
    }
}

#[derive(Debug, Deserialize)]
struct EvaluationPayload {
    decision: String,
    feedback: String,
}

enum EvaluationDecision {
    Pass,
    RetryStep,
    Replan,
    Fail,
}

impl EvaluationDecision {
    fn from_label(label: &str) -> Self {
        match label {
            "pass" => Self::Pass,
            "retry_step" => Self::RetryStep,
            "replan" => Self::Replan,
            _ => Self::Fail,
        }
    }
}

fn step_result_from_rag(step: &PlanStep, rag_response: RagResponse) -> StepResult {
    StepResult {
        step_id: step.id.clone(),
        summary: summarize(&rag_response.answer),
        raw_text: rag_response.answer,
        tool_trace: None,
        citations: rag_response.cited_chunk_ids,
    }
}

fn response_text_or_json(response: &GenerateContentResponse) -> String {
    extract_text_response(response)
        .unwrap_or_else(|| serde_json::to_string(response).unwrap_or_default())
}

fn summarize(text: &str) -> String {
    const LIMIT: usize = 160;
    let mut summary = text.trim().chars().take(LIMIT).collect::<String>();
    if text.trim().chars().count() > LIMIT {
        summary.push_str("...");
    }
    summary
}

impl From<GeminiError> for PlanningError {
    fn from(error: GeminiError) -> Self {
        Self::Backend(Box::new(error))
    }
}

impl From<RagError> for PlanningError {
    fn from(error: RagError) -> Self {
        Self::Rag(Box::new(error))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{PlanOutcome, PlanningConfig, PlanningError, PlanningSession};
    use crate::{
        agentic::{
            test_support::{response_with_text, ScriptedBackend, StaticRetriever},
            tool_runtime::{AgentTools, ToolRegistry},
        },
        types::{FunctionDeclaration, Tool, ToolConfigFunctionDeclaration},
        FunctionHandler,
    };

    fn build_tools() -> AgentTools {
        let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
            function_declarations: vec![FunctionDeclaration {
                name: "lookup_status".to_string(),
                description: "Looks up a status".to_string(),
                parameters: None,
                parameters_json_schema: None,
                response: None,
            }],
        });
        let mut handlers = ToolRegistry::new();
        handlers.insert(
            "lookup_status".to_string(),
            FunctionHandler::Sync(Box::new(|_| Ok(json!({"status": "green"})))),
        );
        AgentTools::new(vec![tool], handlers)
    }

    #[tokio::test]
    async fn planning_runs_steps_sequentially_and_synthesizes() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s1","title":"Inspect","instruction":"Inspect status","success_criteria":"Have status","allowed_tools":["lookup_status"],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|request| {
                assert_eq!(request.contents.len(), 1);
                Ok(response_with_text("Status captured"))
            }),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"pass","feedback":"Looks good"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("Final answer"))),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let outcome = session
            .run(
                "gemini-test",
                "Find the system status",
                Some(&build_tools()),
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("planning should succeed");

        assert_eq!(outcome.final_answer, "Final answer");
        assert_eq!(outcome.trace.step_results.len(), 1);
    }

    #[tokio::test]
    async fn planning_retries_each_step_only_once() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s1","title":"Inspect","instruction":"Inspect status","success_criteria":"Have status","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("first try"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"retry_step","feedback":"Try again"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("second try"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"retry_step","feedback":"Still bad"}"#,
                ))
            }),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let error = session
            .run(
                "gemini-test",
                "Find the system status",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect_err("retry limit should be enforced");

        assert!(matches!(
            error,
            PlanningError::RetryLimitExceeded { step_id } if step_id == "s1"
        ));
    }

    #[tokio::test]
    async fn planning_allows_a_single_replan() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s1","title":"Inspect","instruction":"Inspect","success_criteria":"Inspect","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("first run"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"replan","feedback":"Need a better plan"}"#,
                ))
            }),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s2","title":"Retry","instruction":"Retry","success_criteria":"Retry","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("second run"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"pass","feedback":"Done"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("Final answer"))),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let outcome = session
            .run(
                "gemini-test",
                "Find the system status",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("single replan should succeed");

        assert_eq!(outcome.trace.replans.len(), 1);
        assert!(outcome.working_memory.entries.contains_key("s2"));
    }

    #[tokio::test]
    async fn planning_exposes_working_memory_to_later_steps() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[
                        {"id":"s1","title":"Step 1","instruction":"Collect data","success_criteria":"Have data","allowed_tools":[],"needs_rag":false},
                        {"id":"s2","title":"Step 2","instruction":"Use prior data","success_criteria":"Use data","allowed_tools":[],"needs_rag":false}
                    ]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("first result"))),
            Box::new(|_| Ok(response_with_text(r#"{"decision":"pass","feedback":"ok"}"#))),
            Box::new(|request| {
                let prompt = &request.contents[0].parts[0];
                let data = serde_json::to_string(prompt).unwrap();
                assert!(data.contains("s1"));
                Ok(response_with_text("second result"))
            }),
            Box::new(|_| Ok(response_with_text(r#"{"decision":"pass","feedback":"ok"}"#))),
            Box::new(|_| Ok(response_with_text("Final answer"))),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let outcome: PlanOutcome = session
            .run(
                "gemini-test",
                "Find the system status",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("planning should succeed");

        assert_eq!(outcome.trace.step_results.len(), 2);
    }

    #[tokio::test]
    async fn planning_requires_retrievers_for_rag_steps() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_text(
                r#"{"steps":[{"id":"s1","title":"Research","instruction":"Research","success_criteria":"Have facts","allowed_tools":[],"needs_rag":true}]}"#,
            ))
        })]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let error = session
            .run(
                "gemini-test",
                "Research something",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect_err("rag steps require a retriever");

        assert!(matches!(
            error,
            PlanningError::RagRequiredButUnavailable { step_id } if step_id == "s1"
        ));
    }
}

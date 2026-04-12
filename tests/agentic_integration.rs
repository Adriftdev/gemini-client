use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
};

use async_trait::async_trait;
use gemini_client_rs::{
    agentic::{
        multi_agent::{SupervisorConfig, SupervisorWorkflow},
        planning::{PlanningConfig, PlanningSession},
        rag::{RagError, RagQuery, RetrievedChunk, Retriever},
        tool_runtime::{AgentTools, ModelBackend},
    },
    types::{
        FunctionDeclaration, FunctionParameters, GenerateContentRequest, GenerateContentResponse,
        ParameterProperty, ParameterPropertyString, Tool, ToolConfigFunctionDeclaration,
    },
    FunctionHandler, GeminiError,
};
use serde_json::json;

type ScriptedResponse = Box<
    dyn Fn(&GenerateContentRequest) -> Result<GenerateContentResponse, GeminiError> + Send + Sync,
>;

struct ScriptedBackend {
    responses: Mutex<VecDeque<ScriptedResponse>>,
}

impl ScriptedBackend {
    fn new(responses: Vec<ScriptedResponse>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::from(responses)),
        }
    }
}

#[async_trait]
impl ModelBackend for ScriptedBackend {
    async fn generate_content(
        &self,
        _model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let response = self
            .responses
            .lock()
            .expect("lock scripted responses")
            .pop_front()
            .ok_or_else(|| GeminiError::Api(json!({"error": "missing scripted response"})))?;

        response(request)
    }
}

struct MemoryRetriever {
    chunks: Vec<RetrievedChunk>,
}

#[async_trait]
impl Retriever for MemoryRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(self.chunks.iter().take(query.top_k).cloned().collect())
    }
}

fn response_with_text(text: &str) -> GenerateContentResponse {
    serde_json::from_value(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{ "text": text }]
            }
        }],
        "usageMetadata": {}
    }))
    .expect("valid response payload")
}

fn build_tools() -> AgentTools {
    let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
        function_declarations: vec![FunctionDeclaration {
            name: "lookup_status".to_string(),
            description: "Looks up a status".to_string(),
            parameters: Some(FunctionParameters {
                parameter_type: "object".to_string(),
                properties: HashMap::from([(
                    "service".to_string(),
                    ParameterProperty::String(ParameterPropertyString {
                        description: Some("Service name".to_string()),
                        enum_values: None,
                    }),
                )]),
                required: Some(vec!["service".to_string()]),
            }),
            parameters_json_schema: None,
            response: None,
        }],
    });
    let mut handlers = HashMap::new();
    handlers.insert(
        "lookup_status".to_string(),
        FunctionHandler::Sync(Box::new(|args| {
            Ok(json!({"status": format!("{} is healthy", args["service"])}))
        })),
    );

    AgentTools::new(vec![tool], handlers)
}

#[tokio::test]
async fn planning_runs_end_to_end_with_mock_backend() {
    let backend = ScriptedBackend::new(vec![
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"steps":[
                    {"id":"research","title":"Research facts","instruction":"Collect facts","success_criteria":"Have cited facts","allowed_tools":[],"needs_rag":true},
                    {"id":"status","title":"Check status","instruction":"Check live status","success_criteria":"Have the latest status","allowed_tools":["lookup_status"],"needs_rag":false}
                ]}"#,
            ))
        }),
        Box::new(|request| {
            assert_eq!(
                request
                    .generation_config
                    .as_ref()
                    .and_then(|config| config.candidate_count),
                Some(1)
            );
            Ok(response_with_text(
                r#"{"answer":"Fact from the corpus","citation_chunk_ids":["doc-1"]}"#,
            ))
        }),
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"decision":"pass","feedback":"research ok"}"#,
            ))
        }),
        Box::new(|_| {
            Ok(serde_json::from_value(json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "lookup_status",
                                "args": { "service": "api" }
                            }
                        }]
                    }
                }],
                "usageMetadata": {}
            }))
            .expect("valid tool-call response"))
        }),
        Box::new(|request| {
            assert!(request.contents.len() >= 3);
            Ok(response_with_text("Status confirmed"))
        }),
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"decision":"pass","feedback":"status ok"}"#,
            ))
        }),
        Box::new(|_| Ok(response_with_text("Final integrated answer"))),
    ]);
    let retriever = MemoryRetriever {
        chunks: vec![RetrievedChunk {
            id: "doc-1".to_string(),
            source: "memory".to_string(),
            title: "Doc 1".to_string(),
            content: "Important fact".to_string(),
            score: 1.0,
            metadata: None,
        }],
    };
    let tools = build_tools();
    let session = PlanningSession::new(&backend, PlanningConfig::default());

    let outcome = session
        .run(
            "gemini-test",
            "Prepare a report",
            Some(&tools),
            Some(&retriever),
        )
        .await
        .expect("planning flow should succeed");

    assert!(outcome.final_answer.contains("Final integrated answer"));
    assert_eq!(outcome.trace.step_results.len(), 2);
    assert_eq!(
        outcome
            .working_memory
            .entries
            .get("research")
            .and_then(|value| value.get("citations"))
            .and_then(|value| value.as_array())
            .map(|value| value.len()),
        Some(1)
    );
}

#[tokio::test]
async fn supervisor_runs_end_to_end_with_mock_backend() {
    let backend = ScriptedBackend::new(vec![
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"assignments":[
                    {"agent_role":"worker","task":"Research the corpus","success_criteria":"Have cited notes","allowed_tools":[],"needs_rag":true},
                    {"agent_role":"worker","task":"Check live status","success_criteria":"Have live status","allowed_tools":["lookup_status"],"needs_rag":false}
                ]}"#,
            ))
        }),
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"answer":"Corpus findings","citation_chunk_ids":["doc-1"]}"#,
            ))
        }),
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"decision":"accept","feedback":"good"}"#,
            ))
        }),
        Box::new(|_| {
            Ok(serde_json::from_value(json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "functionCall": {
                                "name": "lookup_status",
                                "args": { "service": "api" }
                            }
                        }]
                    }
                }],
                "usageMetadata": {}
            }))
            .expect("valid tool-call response"))
        }),
        Box::new(|_| Ok(response_with_text("Status findings"))),
        Box::new(|_| {
            Ok(response_with_text(
                r#"{"decision":"accept","feedback":"good"}"#,
            ))
        }),
        Box::new(|_| Ok(response_with_text("Supervisor final answer"))),
    ]);
    let retriever = MemoryRetriever {
        chunks: vec![RetrievedChunk {
            id: "doc-1".to_string(),
            source: "memory".to_string(),
            title: "Doc 1".to_string(),
            content: "Important fact".to_string(),
            score: 1.0,
            metadata: None,
        }],
    };
    let tools = build_tools();
    let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());

    let outcome = workflow
        .run(
            "gemini-test",
            "Prepare a report",
            Some(&tools),
            Some(&retriever),
        )
        .await
        .expect("supervisor workflow should succeed");

    assert_eq!(outcome.assignments.len(), 2);
    assert_eq!(outcome.accepted_artifacts.len(), 2);
    assert!(outcome.final_answer.contains("Supervisor final answer"));
}

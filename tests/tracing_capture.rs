#![cfg(feature = "tracing")]

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    future::Future,
    io::{Read, Write},
    net::TcpListener,
    sync::{Arc, Mutex},
    thread,
};

use async_trait::async_trait;
use gemini_client_rs::{
    agentic::{
        multi_agent::{SupervisorConfig, SupervisorWorkflow},
        planning::{PlanningConfig, PlanningSession},
        rag::{RagConfig, RagError, RagQuery, RagSession, RetrievedChunk, Retriever},
        tool_runtime::{
            execute_tool_loop, ModelBackend, ToolRegistry, ToolRegistryView, ToolRuntimeConfig,
            Toolbox,
        },
    },
    types::{
        Content, ContentPart, FunctionDeclaration, FunctionParameters, GenerateContentRequest,
        GenerateContentResponse, ParameterProperty, ParameterPropertyString, Role, Tool,
        ToolConfigFunctionDeclaration,
    },
    FunctionHandler, GeminiClient, GeminiError,
};
use serde_json::{json, Value};
use tracing::{field::Field, Event, Subscriber};
use tracing_subscriber::{
    layer::{Context, Layer},
    prelude::*,
    registry::LookupSpan,
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum RecordKind {
    Span,
    Event,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TraceRecord {
    kind: RecordKind,
    name: String,
    level: Option<String>,
    fields: BTreeMap<String, String>,
}

impl TraceRecord {
    fn has_field(&self, key: &str, value: &str) -> bool {
        self.fields.get(key).map(String::as_str) == Some(value)
    }
}

#[derive(Clone, Default)]
struct CaptureLayer {
    records: Arc<Mutex<Vec<TraceRecord>>>,
}

impl CaptureLayer {
    fn new(records: Arc<Mutex<Vec<TraceRecord>>>) -> Self {
        Self { records }
    }

    fn push(&self, record: TraceRecord) {
        self.records
            .lock()
            .expect("trace records lock")
            .push(record);
    }
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: Context<'_, S>,
    ) {
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        self.push(TraceRecord {
            kind: RecordKind::Span,
            name: attrs.metadata().name().to_string(),
            level: Some(attrs.metadata().level().to_string()),
            fields: visitor.fields,
        });
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        self.push(TraceRecord {
            kind: RecordKind::Event,
            name: event.metadata().target().to_string(),
            level: Some(event.metadata().level().to_string()),
            fields: visitor.fields,
        });
    }
}

#[derive(Default)]
struct FieldVisitor {
    fields: BTreeMap<String, String>,
}

impl tracing::field::Visit for FieldVisitor {
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

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
            .expect("scripted backend lock")
            .pop_front()
            .expect("scripted response should exist");
        response(request)
    }
}

struct StaticRetriever {
    chunks: Vec<RetrievedChunk>,
}

#[async_trait]
impl Retriever for StaticRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(self.chunks.iter().take(query.top_k).cloned().collect())
    }
}

fn capture_records<T, F>(future: F) -> (T, Vec<TraceRecord>)
where
    F: Future<Output = T>,
{
    let records = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(CaptureLayer::new(records.clone()));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let output = tracing::subscriber::with_default(subscriber, || runtime.block_on(future));
    let captured = records.lock().expect("trace records lock").clone();
    (output, captured)
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
    .expect("valid response")
}

fn response_with_function_calls(parts: Vec<Value>) -> GenerateContentResponse {
    serde_json::from_value(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts
            }
        }],
        "usageMetadata": {}
    }))
    .expect("valid response")
}

fn build_toolbox() -> (Toolbox, ToolRegistry) {
    let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
        function_declarations: vec![FunctionDeclaration {
            name: "lookup_status".to_string(),
            description: "Looks up service status".to_string(),
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
    let mut handlers = ToolRegistry::new();
    handlers.insert(
        "lookup_status".to_string(),
        FunctionHandler::Sync(Box::new(|args| {
            Ok(json!({"status": format!("{} is healthy", args["service"])}))
        })),
    );

    (Toolbox::new(vec![tool]), handlers)
}

fn spawn_http_server(responses: Vec<String>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let address = listener.local_addr().expect("local address");
    let handle = thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().expect("accept connection");
            let mut buffer = [0u8; 8192];
            let _ = stream.read(&mut buffer);
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            stream.flush().expect("flush response");
        }
    });

    (format!("http://{address}"), handle)
}

fn http_json_response(status_line: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn http_sse_response(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn assert_has_span(records: &[TraceRecord], name: &str) {
    assert!(
        records
            .iter()
            .any(|record| record.kind == RecordKind::Span && record.name == name),
        "missing span {name}; records: {records:?}"
    );
}

fn assert_has_event_field(records: &[TraceRecord], key: &str, value: &str) {
    assert!(
        records
            .iter()
            .any(|record| record.kind == RecordKind::Event && record.has_field(key, value)),
        "missing event field {key}={value}; records: {records:?}"
    );
}

#[test]
fn low_level_client_emits_generate_and_stream_traces() {
    let (_, records) = capture_records(async {
        let request = serde_json::from_value::<GenerateContentRequest>(json!({
            "contents": [{
                "role": "user",
                "parts": [{ "text": "Check release status" }]
            }]
        }))
        .expect("request");

        let success_body = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "healthy" }]
                }
            }],
            "usageMetadata": {}
        })
        .to_string();
        let (api_url, handle) =
            spawn_http_server(vec![http_json_response("200 OK", &success_body)]);
        let client = GeminiClient::new("test-key".to_string()).with_api_url(api_url);
        let response = client
            .generate_content("gemini-test", &request)
            .await
            .expect("generate content should succeed");
        assert_eq!(response.candidates.len(), 1);
        handle.join().expect("join server");

        let error_body = json!({
            "error": {
                "code": 400,
                "message": "bad request",
                "status": "INVALID_ARGUMENT"
            }
        })
        .to_string();
        let (api_url, handle) =
            spawn_http_server(vec![http_json_response("400 Bad Request", &error_body)]);
        let client = GeminiClient::new("test-key".to_string()).with_api_url(api_url);
        let error = client
            .generate_content("gemini-test", &request)
            .await
            .expect_err("generate content failure should be surfaced");
        assert!(matches!(error, GeminiError::Api(_)));
        handle.join().expect("join server");

        let success_stream_body = format!(
            "data: {}\n\n",
            json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{ "text": "streamed" }]
                    }
                }],
                "usageMetadata": {}
            })
        );
        let (api_url, handle) = spawn_http_server(vec![http_sse_response(&success_stream_body)]);
        let client = GeminiClient::new("test-key".to_string()).with_api_url(api_url);
        let stream = client
            .stream_content("gemini-test", &request)
            .await
            .expect("stream should start");
        futures_util::pin_mut!(stream);
        let first = futures_util::StreamExt::next(&mut stream)
            .await
            .expect("first streamed message")
            .expect("streamed payload should parse");
        assert_eq!(first.candidates.len(), 1);
        handle.join().expect("join server");

        let invalid_stream_body = "data: not-json\n\n".to_string();
        let (api_url, handle) = spawn_http_server(vec![http_sse_response(&invalid_stream_body)]);
        let client = GeminiClient::new("test-key".to_string()).with_api_url(api_url);
        let stream = client
            .stream_content("gemini-test", &request)
            .await
            .expect("stream should start");
        futures_util::pin_mut!(stream);
        let error = futures_util::StreamExt::next(&mut stream)
            .await
            .expect("first streamed message")
            .expect_err("invalid json should surface");
        assert!(matches!(error, GeminiError::Json { .. }));
        handle.join().expect("join server");
    });

    assert_has_span(&records, "gemini_client_rs.generate_content");
    assert_has_span(&records, "gemini_client_rs.stream_content");
    assert_has_event_field(&records, "candidate_count", "1");
    assert_has_event_field(&records, "error_kind", "api");
    assert_has_event_field(&records, "error_kind", "json");
}

#[test]
fn tool_runtime_emits_tool_loop_traces() {
    let (_, records) = capture_records(async {
        let request = GenerateContentRequest {
            system_instruction: None,
            contents: vec![Content {
                parts: vec![ContentPart::new_text("Check api", false)],
                role: Some(Role::User),
            }],
            tools: vec![],
            tool_config: None,
            generation_config: None,
        };

        let (toolbox, handlers) = build_toolbox();
        let selection = ToolRegistryView::all(&toolbox, &handlers);
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_function_calls(vec![json!({
                    "functionCall": {
                        "name": "lookup_status",
                        "args": { "service": "api" }
                    }
                })]))
            }),
            Box::new(|_| Ok(response_with_text("done"))),
        ]);
        let _ = execute_tool_loop(
            &backend,
            "gemini-test",
            request.clone(),
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect("successful tool loop");

        let empty_toolbox = Toolbox::empty();
        let empty_handlers = ToolRegistry::new();
        let selection = ToolRegistryView::all(&empty_toolbox, &empty_handlers);
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_function_calls(vec![json!({
                "functionCall": {
                    "name": "missing_tool",
                    "args": {}
                }
            })]))
        })]);
        let _ = execute_tool_loop(
            &backend,
            "gemini-test",
            request.clone(),
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect_err("unknown tool should fail");

        let (_, handlers) = build_toolbox();
        let toolbox = Toolbox::empty();
        let selection = ToolRegistryView::all(&toolbox, &handlers);
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_function_calls(vec![json!({
                "functionCall": {
                    "name": "lookup_status",
                    "args": { "service": "api" }
                }
            })]))
        })]);
        let _ = execute_tool_loop(
            &backend,
            "gemini-test",
            request,
            Some(&selection),
            &ToolRuntimeConfig {
                max_round_trips: 1,
                allow_parallel_calls: false,
            },
        )
        .await
        .expect_err("loop limit should fail");
    });

    assert_has_span(&records, "gemini_client_rs.tool_loop");
    assert_has_event_field(&records, "tool_name", "lookup_status");
    assert_has_event_field(&records, "error_kind", "function_execution");
    assert_has_event_field(&records, "error_kind", "loop_limit_exceeded");
}

#[test]
fn rag_emits_success_and_warning_traces() {
    let (_, records) = capture_records(async {
        let retriever = StaticRetriever {
            chunks: vec![RetrievedChunk {
                id: "doc-1".to_string(),
                source: "memory".to_string(),
                title: "Doc 1".to_string(),
                content: "Important fact".to_string(),
                score: 1.0,
                metadata: None,
            }],
        };
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_text(
                r#"{"answer":"Done","citation_chunk_ids":["doc-1"]}"#,
            ))
        })]);
        let session = RagSession::new(&backend, &retriever, RagConfig::default());
        let _ = session
            .answer("gemini-test", "What is the fact?", Some("Use citations"))
            .await
            .expect("rag success");

        let empty_retriever = StaticRetriever { chunks: vec![] };
        let backend = ScriptedBackend::new(vec![]);
        let session = RagSession::new(&backend, &empty_retriever, RagConfig::default());
        let _ = session
            .answer("gemini-test", "What is the fact?", None)
            .await
            .expect_err("empty retrieval should fail");

        let retriever = StaticRetriever {
            chunks: vec![RetrievedChunk {
                id: "doc-1".to_string(),
                source: "memory".to_string(),
                title: "Doc 1".to_string(),
                content: "Important fact".to_string(),
                score: 1.0,
                metadata: None,
            }],
        };
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_text(
                r#"{"answer":"Done","citation_chunk_ids":["missing"]}"#,
            ))
        })]);
        let session = RagSession::new(&backend, &retriever, RagConfig::default());
        let _ = session
            .answer("gemini-test", "What is the fact?", None)
            .await
            .expect_err("invalid citations should fail");
    });

    assert_has_span(&records, "gemini_client_rs.rag.answer");
    assert_has_event_field(&records, "retrieval_count", "1");
    assert_has_event_field(&records, "citation_count", "1");
    assert_has_event_field(&records, "invalid_citation_count", "1");
}

#[test]
fn planning_emits_pass_retry_and_replan_traces() {
    let (_, records) = capture_records(async {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s1","title":"Inspect","instruction":"Inspect","success_criteria":"Have status","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("step result"))),
            Box::new(|_| Ok(response_with_text(r#"{"decision":"pass","feedback":"ok"}"#))),
            Box::new(|_| Ok(response_with_text("final answer"))),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let _ = session
            .run(
                "gemini-test",
                "Check the status",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("planning success");

        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s1","title":"Inspect","instruction":"Inspect","success_criteria":"Have status","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("step result"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"retry_step","feedback":"try again"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("step result again"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"retry_step","feedback":"still bad"}"#,
                ))
            }),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let _ = session
            .run(
                "gemini-test",
                "Check the status",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect_err("retry limit should fail");

        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s1","title":"Inspect","instruction":"Inspect","success_criteria":"Inspect","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("step result"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"replan","feedback":"redo"}"#,
                ))
            }),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"steps":[{"id":"s2","title":"Retry","instruction":"Retry","success_criteria":"Retry","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("retry result"))),
            Box::new(|_| Ok(response_with_text(r#"{"decision":"pass","feedback":"ok"}"#))),
            Box::new(|_| Ok(response_with_text("final answer"))),
        ]);
        let session = PlanningSession::new(&backend, PlanningConfig::default());
        let _ = session
            .run(
                "gemini-test",
                "Check the status",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("replan success");
    });

    assert_has_span(&records, "gemini_client_rs.plan.run");
    assert_has_span(&records, "gemini_client_rs.plan.step");
    assert_has_span(&records, "gemini_client_rs.plan.evaluate");
    assert_has_event_field(&records, "step_count", "1");
    assert_has_event_field(&records, "decision", "retry_step");
    assert_has_event_field(&records, "replan_count", "1");
}

#[test]
fn supervisor_emits_accept_revise_and_no_artifact_traces() {
    let (_, records) = capture_records(async {
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
        let _ = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("supervisor success");

        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"assignments":[{"agent_role":"worker","task":"one","success_criteria":"done","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"revise","feedback":"tighten"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact revised"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"accept","feedback":"good"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("final"))),
        ]);
        let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());
        let _ = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect("supervisor revision success");

        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"assignments":[{"agent_role":"worker","task":"one","success_criteria":"done","allowed_tools":[],"needs_rag":false}]}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact one"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"revise","feedback":"tighten"}"#,
                ))
            }),
            Box::new(|_| Ok(response_with_text("artifact revised"))),
            Box::new(|_| {
                Ok(response_with_text(
                    r#"{"decision":"revise","feedback":"still bad"}"#,
                ))
            }),
        ]);
        let workflow = SupervisorWorkflow::new(&backend, SupervisorConfig::default());
        let _ = workflow
            .run(
                "gemini-test",
                "Prepare a report",
                None,
                Option::<&StaticRetriever>::None,
            )
            .await
            .expect_err("supervisor should fail without accepted artifacts");
    });

    assert_has_span(&records, "gemini_client_rs.supervisor.run");
    assert_has_span(&records, "gemini_client_rs.supervisor.assignment");
    assert_has_span(&records, "gemini_client_rs.supervisor.review");
    assert_has_event_field(&records, "assignments_count", "1");
    assert_has_event_field(&records, "decision", "revise");
    assert_has_event_field(&records, "accepted_artifacts_count", "1");
}

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    types::{
        Content, ContentData, FunctionResponse, FunctionResponsePayload, GenerateContentRequest,
        GenerateContentResponse, Tool, ToolConfigFunctionDeclaration,
    },
    FunctionHandler, GeminiClient, GeminiError,
};

pub type ToolRegistry = HashMap<String, FunctionHandler>;

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn generate_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError>;
}

#[async_trait]
impl ModelBackend for GeminiClient {
    async fn generate_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        GeminiClient::generate_content(self, model, request).await
    }
}

#[derive(Debug, Clone)]
pub struct ToolRuntimeConfig {
    pub max_round_trips: usize,
    pub allow_parallel_calls: bool,
}

impl Default for ToolRuntimeConfig {
    fn default() -> Self {
        Self {
            max_round_trips: 8,
            allow_parallel_calls: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRecord {
    pub round_trip: usize,
    pub id: Option<String>,
    pub name: String,
    pub arguments: Value,
    pub response: Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ToolTrace {
    pub round_trips: usize,
    pub calls: Vec<ToolCallRecord>,
}

#[derive(Debug, Clone)]
pub struct ToolRunResult {
    pub response: GenerateContentResponse,
    pub trace: ToolTrace,
}

#[derive(Debug, Clone)]
pub struct Toolbox {
    tools: Vec<Tool>,
}

impl Toolbox {
    pub fn new(tools: Vec<Tool>) -> Self {
        Self { tools }
    }

    pub fn empty() -> Self {
        Self { tools: vec![] }
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn available_tool_names(&self, handlers: &ToolRegistry) -> Vec<String> {
        let mut names = self
            .tools
            .iter()
            .flat_map(extract_tool_names)
            .collect::<Vec<_>>();

        for name in handlers.keys() {
            if !names.iter().any(|existing| existing == name) {
                names.push(name.clone());
            }
        }

        names.sort();
        names
    }

    pub fn missing_allowed_names(
        &self,
        handlers: &ToolRegistry,
        allowed_names: &[String],
    ) -> Vec<String> {
        let known = self
            .available_tool_names(handlers)
            .into_iter()
            .collect::<HashSet<_>>();

        allowed_names
            .iter()
            .filter(|name| !known.contains(*name))
            .cloned()
            .collect()
    }

    pub fn select_tools(&self, allowed_names: &[String]) -> Vec<Tool> {
        if allowed_names.is_empty() {
            return vec![];
        }

        let allowed = allowed_names.iter().cloned().collect::<HashSet<_>>();
        self.tools
            .iter()
            .filter_map(|tool| match tool {
                Tool::FunctionDeclaration(config) => {
                    let function_declarations = config
                        .function_declarations
                        .iter()
                        .filter(|declaration| allowed.contains(&declaration.name))
                        .cloned()
                        .collect::<Vec<_>>();

                    if function_declarations.is_empty() {
                        None
                    } else {
                        Some(Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
                            function_declarations,
                        }))
                    }
                }
                Tool::DynamicRetrieval {
                    google_search_retrieval,
                } if allowed.contains("google_search_retrieval") => Some(Tool::DynamicRetrieval {
                    google_search_retrieval: google_search_retrieval.clone(),
                }),
                Tool::GoogleSearch { google_search } if allowed.contains("google_search") => {
                    Some(Tool::GoogleSearch {
                        google_search: google_search.clone(),
                    })
                }
                Tool::UrlContext { url_context } if allowed.contains("url_context") => {
                    Some(Tool::UrlContext {
                        url_context: url_context.clone(),
                    })
                }
                Tool::CodeExecution { code_execution } if allowed.contains("code_execution") => {
                    Some(Tool::CodeExecution {
                        code_execution: code_execution.clone(),
                    })
                }
                _ => None,
            })
            .collect()
    }

    pub fn filter_handlers<'a>(
        &'a self,
        handlers: &'a ToolRegistry,
        allowed_names: &[String],
    ) -> ToolRegistryView<'a> {
        if allowed_names.is_empty() {
            return ToolRegistryView::empty(self.tools.clone());
        }

        let allowed = allowed_names.iter().cloned().collect::<HashSet<_>>();
        let filtered_handlers = handlers
            .iter()
            .filter(|(name, _)| allowed.contains(*name))
            .map(|(name, handler)| (name.as_str(), handler))
            .collect::<HashMap<_, _>>();

        ToolRegistryView {
            tools: self.select_tools(allowed_names),
            handlers: filtered_handlers,
        }
    }
}

pub struct AgentTools {
    pub toolbox: Toolbox,
    pub handlers: ToolRegistry,
}

impl AgentTools {
    pub fn new(tools: Vec<Tool>, handlers: ToolRegistry) -> Self {
        Self {
            toolbox: Toolbox::new(tools),
            handlers,
        }
    }

    pub fn available_tool_names(&self) -> Vec<String> {
        self.toolbox.available_tool_names(&self.handlers)
    }

    pub fn missing_allowed_names(&self, allowed_names: &[String]) -> Vec<String> {
        self.toolbox
            .missing_allowed_names(&self.handlers, allowed_names)
    }

    pub fn all(&self) -> ToolRegistryView<'_> {
        ToolRegistryView::all(&self.toolbox, &self.handlers)
    }

    pub fn select(&self, allowed_names: &[String]) -> ToolRegistryView<'_> {
        self.toolbox.filter_handlers(&self.handlers, allowed_names)
    }
}

pub struct ToolRegistryView<'a> {
    tools: Vec<Tool>,
    handlers: HashMap<&'a str, &'a FunctionHandler>,
}

impl<'a> ToolRegistryView<'a> {
    pub fn empty(tools: Vec<Tool>) -> Self {
        Self {
            tools,
            handlers: HashMap::new(),
        }
    }

    pub fn all(toolbox: &'a Toolbox, handlers: &'a ToolRegistry) -> Self {
        Self {
            tools: toolbox.tools.clone(),
            handlers: handlers
                .iter()
                .map(|(name, handler)| (name.as_str(), handler))
                .collect(),
        }
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    fn handler(&self, name: &str) -> Option<&FunctionHandler> {
        self.handlers.get(name).copied()
    }
}

pub async fn execute_tool_loop<B: ModelBackend>(
    backend: &B,
    model: &str,
    mut request: GenerateContentRequest,
    tools: Option<&ToolRegistryView<'_>>,
    config: &ToolRuntimeConfig,
) -> Result<ToolRunResult, GeminiError> {
    let _span = crate::telemetry::telemetry_span_guard!(
        info,
        "gemini_client_rs.tool_loop",
        model = model,
        max_round_trips = config.max_round_trips,
        request_tools_count = request.tools.len(),
        has_tool_view = tools.is_some()
    );
    crate::telemetry::telemetry_info!("tool_loop started");

    if request.tools.is_empty() {
        if let Some(tool_view) = tools {
            request.tools = tool_view.tools().to_vec();
            crate::telemetry::telemetry_debug!(
                request_tools_count = request.tools.len(),
                "tool_loop populated request tools from tool view"
            );
        }
    }

    let mut trace = ToolTrace::default();

    for round_trip in 1..=config.max_round_trips {
        crate::telemetry::telemetry_debug!(round_trip, "tool_loop round trip started");
        let response = backend.generate_content(model, &request).await?;

        let Some(candidate) = response.candidates.first() else {
            trace.round_trips = round_trip;
            crate::telemetry::telemetry_info!(
                round_trips = trace.round_trips,
                total_calls = trace.calls.len(),
                "tool_loop completed without candidates"
            );
            return Ok(ToolRunResult { response, trace });
        };

        let Some(content) = candidate.content.clone() else {
            trace.round_trips = round_trip;
            crate::telemetry::telemetry_info!(
                round_trips = trace.round_trips,
                total_calls = trace.calls.len(),
                "tool_loop completed without content"
            );
            return Ok(ToolRunResult { response, trace });
        };

        let function_calls = content
            .parts
            .iter()
            .filter_map(|part| match &part.data {
                ContentData::FunctionCall(function_call) => Some(function_call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        if function_calls.is_empty() {
            trace.round_trips = round_trip;
            crate::telemetry::telemetry_info!(
                round_trips = trace.round_trips,
                total_calls = trace.calls.len(),
                "tool_loop completed without further tool calls"
            );
            return Ok(ToolRunResult { response, trace });
        }

        let Some(tool_view) = tools else {
            let error = GeminiError::FunctionExecution(
                "Model requested tool calls but no tool handlers were provided".to_string(),
            );
            crate::telemetry::telemetry_warn!(
                error_kind = crate::telemetry::gemini_error_kind(&error),
                round_trip,
                "tool_loop missing tool handlers"
            );
            return Err(error);
        };

        request.contents.push(content);

        for function_call in function_calls {
            let Some(handler) = tool_view.handler(&function_call.name) else {
                let error = GeminiError::FunctionExecution(format!(
                    "Unknown function: {}",
                    function_call.name
                ));
                crate::telemetry::telemetry_warn!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    tool_name = function_call.name.as_str(),
                    round_trip,
                    "tool_loop unknown tool requested"
                );
                return Err(error);
            };

            crate::telemetry::telemetry_debug!(
                tool_name = function_call.name.as_str(),
                round_trip,
                "tool_loop executing function call"
            );
            let mut arguments = function_call.arguments.clone();
            let result = handler.execute(&mut arguments).await.map_err(|error| {
                let error = GeminiError::FunctionExecution(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    tool_name = function_call.name.as_str(),
                    round_trip,
                    "tool_loop handler execution failed"
                );
                error
            })?;

            trace.calls.push(ToolCallRecord {
                round_trip,
                id: function_call.id.clone(),
                name: function_call.name.clone(),
                arguments,
                response: result.clone(),
            });

            request.contents.push(Content {
                parts: vec![ContentData::FunctionResponse(FunctionResponse {
                    id: function_call.id.clone(),
                    name: function_call.name.clone(),
                    response: FunctionResponsePayload { content: result },
                })
                .into()],
                role: None,
            });
        }
    }

    let error = GeminiError::LoopLimitExceeded {
        max_round_trips: config.max_round_trips,
    };
    crate::telemetry::telemetry_warn!(
        error_kind = crate::telemetry::gemini_error_kind(&error),
        max_round_trips = config.max_round_trips,
        total_calls = trace.calls.len(),
        "tool_loop exceeded round trip limit"
    );
    Err(error)
}

fn extract_tool_names(tool: &Tool) -> Vec<String> {
    match tool {
        Tool::FunctionDeclaration(config) => config
            .function_declarations
            .iter()
            .map(|declaration| declaration.name.clone())
            .collect(),
        Tool::DynamicRetrieval { .. } => vec!["google_search_retrieval".to_string()],
        Tool::GoogleSearch { .. } => vec!["google_search".to_string()],
        Tool::UrlContext { .. } => vec!["url_context".to_string()],
        Tool::CodeExecution { .. } => vec!["code_execution".to_string()],
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::{execute_tool_loop, ToolRegistry, ToolRegistryView, ToolRuntimeConfig, Toolbox};
    use crate::{
        agentic::test_support::{
            response_with_function_calls, response_with_text, ScriptedBackend,
        },
        types::{
            FunctionDeclaration, FunctionParameters, GenerateContentRequest, ParameterProperty,
            ParameterPropertyString, Tool, ToolConfigFunctionDeclaration,
        },
        FunctionHandler, GeminiError,
    };

    fn weather_toolbox() -> (Toolbox, ToolRegistry) {
        let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
            function_declarations: vec![
                FunctionDeclaration {
                    name: "get_weather".to_string(),
                    description: "Fetches weather".to_string(),
                    parameters: Some(FunctionParameters {
                        parameter_type: "object".to_string(),
                        properties: HashMap::from([(
                            "location".to_string(),
                            ParameterProperty::String(ParameterPropertyString {
                                description: Some("City".to_string()),
                                enum_values: None,
                            }),
                        )]),
                        required: Some(vec!["location".to_string()]),
                    }),
                    parameters_json_schema: None,
                    response: None,
                },
                FunctionDeclaration {
                    name: "get_timezone".to_string(),
                    description: "Fetches timezone".to_string(),
                    parameters: Some(FunctionParameters {
                        parameter_type: "object".to_string(),
                        properties: HashMap::from([(
                            "location".to_string(),
                            ParameterProperty::String(ParameterPropertyString {
                                description: Some("City".to_string()),
                                enum_values: None,
                            }),
                        )]),
                        required: Some(vec!["location".to_string()]),
                    }),
                    parameters_json_schema: None,
                    response: None,
                },
            ],
        });

        let mut handlers = ToolRegistry::new();
        handlers.insert(
            "get_weather".to_string(),
            FunctionHandler::Sync(Box::new(|args| {
                Ok(json!({ "forecast": format!("sunny in {}", args["location"]) }))
            })),
        );
        handlers.insert(
            "get_timezone".to_string(),
            FunctionHandler::Sync(Box::new(|args| {
                Ok(json!({ "timezone": format!("tz for {}", args["location"]) }))
            })),
        );

        (Toolbox::new(vec![tool]), handlers)
    }

    #[tokio::test]
    async fn executes_multiple_tool_calls_from_candidate_zero() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|request| {
                assert_eq!(request.contents.len(), 1);
                Ok(response_with_function_calls(vec![
                    json!({"functionCall": {"id": "1", "name": "get_weather", "args": {"location": "London"}}}),
                    json!({"functionCall": {"id": "2", "name": "get_timezone", "args": {"location": "London"}}}),
                ]))
            }),
            Box::new(|request| {
                assert_eq!(request.contents.len(), 4);
                Ok(response_with_text("done"))
            }),
        ]);
        let (toolbox, handlers) = weather_toolbox();
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let result = execute_tool_loop(
            &backend,
            "gemini-test",
            GenerateContentRequest {
                system_instruction: None,
                contents: vec![crate::agentic::build_user_content("Help me")],
                tools: vec![],
                tool_config: None,
                generation_config: None,
            },
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect("tool loop should succeed");

        assert_eq!(result.trace.calls.len(), 2);
        assert_eq!(result.trace.round_trips, 2);
    }

    #[tokio::test]
    async fn returns_unknown_tool_error_when_handler_is_missing() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_function_calls(vec![json!({
                "functionCall": {"name": "missing_tool", "args": {}}
            })]))
        })]);
        let toolbox = Toolbox::empty();
        let handlers = ToolRegistry::new();
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let error = execute_tool_loop(
            &backend,
            "gemini-test",
            GenerateContentRequest {
                system_instruction: None,
                contents: vec![crate::agentic::build_user_content("Help me")],
                tools: vec![],
                tool_config: None,
                generation_config: None,
            },
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect_err("unknown tool should fail");

        assert!(matches!(
            error,
            GeminiError::FunctionExecution(message) if message.contains("missing_tool")
        ));
    }

    #[tokio::test]
    async fn propagates_handler_failures() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_function_calls(vec![json!({
                "functionCall": {"name": "get_weather", "args": {"location": "London"}}
            })]))
        })]);
        let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
            function_declarations: vec![FunctionDeclaration {
                name: "get_weather".to_string(),
                description: "Fetches weather".to_string(),
                parameters: None,
                parameters_json_schema: None,
                response: None,
            }],
        });
        let mut handlers = ToolRegistry::new();
        handlers.insert(
            "get_weather".to_string(),
            FunctionHandler::Sync(Box::new(|_| Err("upstream failed".to_string()))),
        );
        let toolbox = Toolbox::new(vec![tool]);
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let error = execute_tool_loop(
            &backend,
            "gemini-test",
            GenerateContentRequest {
                system_instruction: None,
                contents: vec![crate::agentic::build_user_content("Help me")],
                tools: vec![],
                tool_config: None,
                generation_config: None,
            },
            Some(&selection),
            &ToolRuntimeConfig::default(),
        )
        .await
        .expect_err("handler failures should propagate");

        assert!(matches!(
            error,
            GeminiError::FunctionExecution(message) if message.contains("upstream failed")
        ));
    }

    #[tokio::test]
    async fn enforces_the_tool_loop_limit() {
        let backend = ScriptedBackend::new(vec![
            Box::new(|_| {
                Ok(response_with_function_calls(vec![json!({
                    "functionCall": {"name": "get_weather", "args": {"location": "London"}}
                })]))
            }),
            Box::new(|_| {
                Ok(response_with_function_calls(vec![json!({
                    "functionCall": {"name": "get_weather", "args": {"location": "London"}}
                })]))
            }),
        ]);
        let (toolbox, handlers) = weather_toolbox();
        let selection = ToolRegistryView::all(&toolbox, &handlers);

        let error = execute_tool_loop(
            &backend,
            "gemini-test",
            GenerateContentRequest {
                system_instruction: None,
                contents: vec![crate::agentic::build_user_content("Help me")],
                tools: vec![],
                tool_config: None,
                generation_config: None,
            },
            Some(&selection),
            &ToolRuntimeConfig {
                max_round_trips: 1,
                allow_parallel_calls: false,
            },
        )
        .await
        .expect_err("loop limit should be enforced");

        assert!(matches!(
            error,
            GeminiError::LoopLimitExceeded { max_round_trips: 1 }
        ));
    }
}

use futures_util::{Stream, StreamExt as _};
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt as _};
use serde_json::Value;
use types::{GenerateContentRequest, GenerateContentResponse};
pub mod agentic;
mod telemetry;
pub mod types;

use anyhow::Result;
use std::future::Future;
use std::pin::Pin; // Using anyhow for cleaner error handling in examples

type SyncFunctionHandler = Box<dyn Fn(&mut Value) -> Result<Value, String> + Send + Sync>;
type AsyncFunctionHandler = Box<
    dyn Fn(&mut Value) -> Pin<Box<dyn Future<Output = Result<Value, String>> + Send>> + Send + Sync,
>;

// The new enum to hold either a sync or async handler
pub enum FunctionHandler {
    Sync(SyncFunctionHandler),
    Async(AsyncFunctionHandler),
}

impl FunctionHandler {
    /// Executes the handler, automatically handling whether it's sync or async.
    pub async fn execute(&self, params: &mut Value) -> Result<Value, String> {
        match self {
            FunctionHandler::Sync(handler) => handler(params),
            FunctionHandler::Async(handler) => handler(params).await,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("HTTP Error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Streaming Event Error: {0}")]
    EventSource(#[from] reqwest_eventsource::Error),
    #[error("API Error: {0}")]
    Api(Value),
    #[error("JSON Error: {error} (payload: {data})")]
    Json {
        data: String,
        #[source]
        error: serde_json::Error,
    },
    #[error("Function execution error: {0}")]
    FunctionExecution(String),
    #[error("Tool loop exceeded the maximum number of round trips ({max_round_trips})")]
    LoopLimitExceeded { max_round_trips: usize },
}

impl GeminiError {
    async fn from_response(
        response: reqwest::Response,
        context: Option<serde_json::Value>,
    ) -> Self {
        let status = response.status();
        let text = match response.text().await {
            Ok(text) => text,
            Err(error) => return Self::Http(error),
        };
        let message = match serde_json::from_str::<Value>(&text) {
            Ok(error) => error,
            Err(_) => serde_json::Value::String(text),
        };

        Self::Api(serde_json::json!({
            "status": status.as_u16(),
            "message": message,
            "context": context.unwrap_or_default(),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct GeminiClient {
    api_key: String,
    http_client: Client,
    api_url: String,
}

impl Default for GeminiClient {
    fn default() -> Self {
        Self {
            api_key: std::env::var("GEMINI_API_KEY").unwrap_or_default(),
            http_client: Client::new(),
            api_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }
}

impl GeminiClient {
    /// Create a new Gemini client.
    ///
    /// If you have the [`GEMINI_API_KEY`] environment variable set, you can use
    /// [`GeminiClient::default()`] instead.
    pub fn new(api_key: String) -> Self {
        GeminiClient {
            api_key,
            ..Default::default()
        }
    }

    /// Provide a pre-configured [`reqwest::Client`] to use for the Gemini
    /// client.
    ///
    /// This can be used to configure things like timeouts, proxies, etc.
    pub fn with_client(mut self, http_client: Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Set the API URL for the Gemini client.
    ///
    /// This is useful for testing purposes.
    pub fn with_api_url(mut self, api_url: String) -> Self {
        self.api_url = api_url;
        self
    }

    /// List all available models.
    pub async fn list_models(&self) -> Result<Vec<types::Model>, GeminiError> {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Response {
            models: Vec<types::Model>,
            next_page_token: Option<String>,
        }

        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.list_models",
            has_api_key = !self.api_key.is_empty()
        );
        crate::telemetry::telemetry_info!("list_models started");

        let mut models = vec![];
        let mut next_page_token = None;
        let mut page_fetch_count = 0usize;
        loop {
            let mut url = format!("{}/models?key={}&pageSize=1000", self.api_url, self.api_key);
            if let Some(ref next_page_token) = next_page_token {
                url.push_str(&format!("&pageToken={next_page_token}"));
            }

            page_fetch_count += 1;
            crate::telemetry::telemetry_debug!(
                page_fetch_count,
                has_page_token = next_page_token.is_some(),
                "list_models fetching page"
            );

            let response = match self.http_client.get(&url).send().await {
                Ok(response) => response,
                Err(error) => {
                    let error = GeminiError::Http(error);
                    crate::telemetry::telemetry_error!(
                        error_kind = crate::telemetry::gemini_error_kind(&error),
                        page_fetch_count,
                        "list_models request failed"
                    );
                    return Err(error);
                }
            };
            if !response.status().is_success() {
                let error = GeminiError::from_response(response, None).await;
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    page_fetch_count,
                    "list_models API failure"
                );
                return Err(error);
            }

            let response: Response = match response.json().await {
                Ok(response) => response,
                Err(error) => {
                    let error = GeminiError::Http(error);
                    crate::telemetry::telemetry_error!(
                        error_kind = crate::telemetry::gemini_error_kind(&error),
                        page_fetch_count,
                        "list_models response parsing failed"
                    );
                    return Err(error);
                }
            };

            models.extend(response.models);
            next_page_token = response.next_page_token;
            if next_page_token.is_none() {
                break;
            }
        }

        let models = models
            .into_iter()
            .map(|mut model| {
                model.base_model_id = model.name.replace("models/", "");
                model
            })
            .collect::<Vec<_>>();
        let _ = page_fetch_count;

        crate::telemetry::telemetry_info!(
            page_fetch_count,
            model_count = models.len(),
            "list_models completed"
        );

        Ok(models)
    }

    pub async fn generate_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.generate_content",
            model,
            contents_count = request.contents.len(),
            tools_count = request.tools.len(),
            has_system_instruction = request.system_instruction.is_some(),
            has_generation_config = request.generation_config.is_some()
        );
        crate::telemetry::telemetry_info!("generate_content started");

        let url = format!(
            "{}/models/{model}:generateContent?key={}",
            self.api_url, self.api_key
        );

        let response = match self.http_client.post(&url).json(request).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = GeminiError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    "generate_content request failed"
                );
                return Err(error);
            }
        };
        if !response.status().is_success() {
            let error = GeminiError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::gemini_error_kind(&error),
                "generate_content API failure"
            );
            return Err(error);
        }

        let response: GenerateContentResponse = match response.json().await {
            Ok(response) => response,
            Err(error) => {
                let error = GeminiError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    "generate_content response parsing failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!(
            candidate_count = response.candidates.len(),
            "generate_content completed"
        );

        Ok(response)
    }

    /// Generates a streamed response from the model given an input
    /// [`GenerateContentRequest`].
    pub async fn stream_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<impl Stream<Item = Result<types::GenerateContentResponse, GeminiError>>, GeminiError>
    {
        let _model_name = model.to_string();
        let _contents_count = request.contents.len();
        let _tools_count = request.tools.len();
        let _has_system_instruction = request.system_instruction.is_some();
        let _has_generation_config = request.generation_config.is_some();
        let url = format!(
            "{}/models/{model}:streamGenerateContent?alt=sse&key={}",
            self.api_url, self.api_key
        );

        let mut stream = self
            .http_client
            .post(&url)
            .json(request)
            .eventsource()
            .expect("can clone request builder");

        Ok(async_stream::stream! {
            let _span = crate::telemetry::telemetry_span_guard!(
                info,
                "gemini_client_rs.stream_content",
                model = _model_name.as_str(),
                contents_count = _contents_count,
                tools_count = _tools_count,
                has_system_instruction = _has_system_instruction,
                has_generation_config = _has_generation_config
            );
            crate::telemetry::telemetry_info!("stream_content started");
            let mut message_count = 0usize;

            while let Some(event) = stream.next().await {
                match event {
                    Ok(event) => match event {
                        Event::Open => crate::telemetry::telemetry_debug!("stream_content opened"),
                        Event::Message(event) => {
                            message_count += 1;
                            crate::telemetry::telemetry_debug!(
                                message_count,
                                "stream_content message received"
                            );
                            yield serde_json::from_str::<types::GenerateContentResponse>(&event.data)
                                .map_err(|error| {
                                    let error = GeminiError::Json {
                                        data: event.data,
                                        error,
                                    };
                                    crate::telemetry::telemetry_error!(
                                        error_kind = crate::telemetry::gemini_error_kind(&error),
                                        message_count,
                                        "stream_content message parsing failed"
                                    );
                                    error
                                })
                        }
                    },
                    Err(e) => match e {
                        reqwest_eventsource::Error::StreamEnded => {
                            crate::telemetry::telemetry_info!(
                                message_count,
                                "stream_content ended"
                            );
                            stream.close()
                        }
                        reqwest_eventsource::Error::InvalidContentType(content_type, response) => {
                            let header = content_type.to_str().unwrap_or_default();
                            let error = GeminiError::from_response(
                                response,
                                Some(serde_json::json!({
                                    "cause": "Invalid content type",
                                    "header": header
                                })),
                            ).await;
                            crate::telemetry::telemetry_error!(
                                error_kind = crate::telemetry::gemini_error_kind(&error),
                                message_count,
                                "stream_content invalid content type"
                            );
                            yield Err(error)
                        }
                        reqwest_eventsource::Error::InvalidStatusCode(_, response) => {
                            let error = GeminiError::from_response(
                                response,
                                Some(serde_json::json!({"cause": "Invalid status code"})),
                            ).await;
                            crate::telemetry::telemetry_error!(
                                error_kind = crate::telemetry::gemini_error_kind(&error),
                                message_count,
                                "stream_content invalid status code"
                            );
                            yield Err(error)
                        }
                        _ => {
                            let error = GeminiError::EventSource(e);
                            crate::telemetry::telemetry_error!(
                                error_kind = crate::telemetry::gemini_error_kind(&error),
                                message_count,
                                "stream_content event source failure"
                            );
                            yield Err(error)
                        }
                    }
                }
            }

            crate::telemetry::telemetry_info!(
                message_count,
                "stream_content completed"
            );
            let _ = message_count;
        })
    }

    pub async fn generate_content_with_function_calling(
        &self,
        model: &str,
        request: GenerateContentRequest,
        function_handlers: &agentic::tool_runtime::ToolRegistry,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.generate_content_with_function_calling",
            model,
            contents_count = request.contents.len(),
            tools_count = request.tools.len(),
            function_handler_count = function_handlers.len()
        );
        crate::telemetry::telemetry_info!("generate_content_with_function_calling started");
        let toolbox = agentic::tool_runtime::Toolbox::empty();
        let tool_view = agentic::tool_runtime::ToolRegistryView::all(&toolbox, function_handlers);
        let result = match agentic::tool_runtime::execute_tool_loop(
            self,
            model,
            request,
            Some(&tool_view),
            &agentic::tool_runtime::ToolRuntimeConfig::default(),
        )
        .await
        {
            Ok(result) => result,
            Err(error) => {
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    function_handler_count = function_handlers.len(),
                    "generate_content_with_function_calling failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!(
            round_trips = result.trace.round_trips,
            tool_call_count = result.trace.calls.len(),
            "generate_content_with_function_calling completed"
        );

        Ok(result.response)
    }
}

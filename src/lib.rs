use std::collections::HashMap;

use futures_util::{Stream, StreamExt as _};
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt as _};
use types::{
    Content, ContentData, FunctionResponse, FunctionResponsePayload, GenerateContentRequest,
    GenerateContentResponse, Role,
};
pub mod types;

pub type FunctionHandler =
    Box<dyn Fn(&mut serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>;

#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("HTTP Error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Streaming Event Error: {0}")]
    EventSource(#[from] reqwest_eventsource::Error),
    #[error("API Error: {0}")]
    Api(String),
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Function execution error: {0}")]
    FunctionExecution(String),
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

        let url = format!("{}/models?key={}&pageSize=1000", self.api_url, self.api_key);

        let response = self.http_client.get(&url).send().await?;
        if !response.status().is_success() {
            return handle_error::<Vec<types::Model>>(response).await;
        }

        let mut models = vec![];
        let mut next_page_token = None;
        loop {
            let mut url = format!("{}/models?key={}&pageSize=1000", self.api_url, self.api_key);
            if let Some(next_page_token) = next_page_token {
                url.push_str(&format!("&pageToken={next_page_token}"));
            }

            let response = self.http_client.get(&url).send().await?;
            if !response.status().is_success() {
                let error_text = response.text().await?;
                return Err(GeminiError::Api(error_text));
            }

            let response: Response = response.json().await?;

            models.extend(response.models);
            next_page_token = response.next_page_token;
            if next_page_token.is_none() {
                break;
            }
        }

        Ok(models
            .into_iter()
            .map(|mut model| {
                model.base_model_id = model.name.replace("models/", "");
                model
            })
            .collect())
    }

    pub async fn generate_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let url = format!(
            "{}/models/{model}:generateContent?key={}",
            self.api_url, self.api_key
        );

        let response = self.http_client.post(&url).json(request).send().await?;
        if !response.status().is_success() {
            return handle_error::<GenerateContentResponse>(response).await;
        }

        Ok(response.json().await?)
    }

    /// Generates a streamed response from the model given an input
    /// [`GenerateContentRequest`].
    pub async fn stream_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<impl Stream<Item = Result<types::GenerateContentResponse, GeminiError>>, GeminiError>
    {
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
            while let Some(event) = stream.next().await {
                match event {
                    Ok(event) => match event {
                        Event::Open => (),
                        Event::Message(event) => yield
                            serde_json::from_str::<types::GenerateContentResponse>(&event.data)
                                .map_err(Into::into),
                    },
                    Err(e) => match e {
                        reqwest_eventsource::Error::StreamEnded => stream.close(),
                        reqwest_eventsource::Error::InvalidContentType(content_type, response) => {
                            let header = content_type.to_str().unwrap_or_default();
                            let body = response.text().await?;
                            yield Err(GeminiError::Api(format!(
                                "Invalid content type {header}: {body}"
                            )))
                        }
                        reqwest_eventsource::Error::InvalidStatusCode(code, response) => {
                            let body = response.text().await?;
                            yield Err(GeminiError::Api(format!(
                                "Invalid status code {code}: {body}"
                            )))
                        }
                        _ => yield Err(e.into()),
                    }
                }
            }
        })
    }

    pub async fn generate_content_with_function_calling(
        &self,
        model: &str,
        mut request: GenerateContentRequest,
        function_handlers: &HashMap<String, FunctionHandler>,
    ) -> Result<GenerateContentResponse, GeminiError> {
        loop {
            let response = self.generate_content(model, &request).await?;

            if let Some(candidate) = response.candidates.first() {
                if let Some(part) = candidate.content.parts.first() {
                    match &part.data {
                        ContentData::Text(_) => return Ok(response),
                        ContentData::FunctionCall(function_call) => {
                            if let Some(handler) = function_handlers.get(&function_call.name) {
                                match handler(&mut function_call.arguments.clone()) {
                                    Ok(result) => {
                                        request.contents.push(Content {
                                            parts: vec![ContentData::FunctionCall(
                                                function_call.clone(),
                                            )
                                            .into()],
                                            role: Role::User,
                                        });

                                        request.contents.push(Content {
                                            parts: vec![ContentData::FunctionResponse(
                                                FunctionResponse {
                                                    name: function_call.name.clone(),
                                                    response: FunctionResponsePayload {
                                                        content: result,
                                                    },
                                                },
                                            )
                                            .into()],
                                            role: Role::Tool,
                                        });
                                    }
                                    Err(e) => return Err(GeminiError::FunctionExecution(e)),
                                }
                            } else {
                                return Err(GeminiError::FunctionExecution(format!(
                                    "Unknown function: {}",
                                    function_call.name
                                )));
                            }
                        }
                        _ => return Ok(response),
                    }
                } else {
                    return Ok(response);
                }
            } else {
                return Ok(response);
            }
        }
    }
}

async fn handle_error<T>(response: reqwest::Response) -> Result<T, GeminiError> {
    let status = response.status();
    let error_text = response.text().await?;

    Err(GeminiError::Api(format!("status {status}: {error_text}")))
}

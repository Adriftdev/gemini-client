use std::collections::HashMap;

use reqwest::Client;
use types::{
    Content, ContentPart, FunctionResponse, FunctionResponsePayload, GenerateContentRequest,
    GenerateContentResponse, PartResponse, Role,
};
pub mod types;

#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("HTTP Error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API Error: {0}")]
    Api(String),
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Function execution error: {0}")]
    FunctionExecution(String),
}

pub struct GeminiClient {
    api_key: String,
    http_client: Client,
    api_url: String,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        GeminiClient {
            api_key,
            http_client: Client::new(),
            api_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        }
    }

    pub async fn generate_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GenerateContentResponse, GeminiError> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.api_url, model, self.api_key
        );

        let response = self.http_client.post(&url).json(request).send().await?;

        if response.status().is_success() {
            let content: GenerateContentResponse = response.json().await?;
            Ok(content)
        } else {
            let error_text = response.text().await?;
            Err(GeminiError::Api(error_text))
        }
    }

    pub async fn generate_content_with_function_calling(
        &self,
        model: &str,
        mut request: GenerateContentRequest,
        function_handlers: &HashMap<
            String,
            Box<dyn Fn(&mut serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
        >,
    ) -> Result<GenerateContentResponse, GeminiError> {
        loop {
            let response = self.generate_content(model, &request).await?;

            if let Some(candidates) = &response.candidates {
                if let Some(candidate) = candidates.first() {
                    if let Some(part) = candidate.content.parts.first() {
                        match part {
                            PartResponse::Text(_) => return Ok(response),
                            PartResponse::FunctionCall(function_call) => {
                                if let Some(handler) = function_handlers.get(&function_call.name) {
                                    match handler(&mut function_call.arguments.clone()) {
                                        Ok(result) => {
                                            request.contents.push(Content {
                                                parts: vec![ContentPart::FunctionCall(
                                                    function_call.clone(),
                                                )],
                                                role: Role::User,
                                            });

                                            request.contents.push(Content {
                                                parts: vec![ContentPart::FunctionResponse(
                                                    FunctionResponse {
                                                        name: function_call.name.clone(),
                                                        response: FunctionResponsePayload {
                                                            content: result,
                                                        },
                                                    },
                                                )],
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
                            PartResponse::FunctionResponse(_) => return Ok(response),
                        }
                    } else {
                        return Ok(response);
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

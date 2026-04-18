use futures_util::{Stream, StreamExt as _};
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt as _};
use serde_json::Value;
use std::pin::Pin;
use types::{
    BatchEmbedContentsRequest, BatchEmbedContentsResponse, EmbedContentRequest,
    EmbedContentResponse, GenerateContentRequest, GenerateContentResponse,
};

mod telemetry;
pub mod types;

pub type GeminiResponseStream =
    Pin<Box<dyn Stream<Item = Result<GenerateContentResponse, GeminiError>> + Send>>;

pub use gemini_client_macros::{gemini_tool, GeminiSchema};

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
    #[deprecated(since = "0.10.0", note = "Use stream_generate_content instead")]
    pub async fn stream_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GeminiResponseStream, GeminiError> {
        self.stream_generate_content(model, request).await
    }

    /// Generates a streamed response from the model given an input
    /// [`GenerateContentRequest`].
    pub async fn stream_generate_content(
        &self,
        model: &str,
        request: &GenerateContentRequest,
    ) -> Result<GeminiResponseStream, GeminiError> {
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

        let stream = async_stream::stream! {
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
        };

        Ok(Box::pin(stream))
    }

    /// Generates embeddings for the provided content.
    pub async fn embed_content(
        &self,
        request: &EmbedContentRequest,
    ) -> Result<EmbedContentResponse, GeminiError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.embed_content",
            model = request.model.as_str(),
            task_type = format!("{:?}", request.task_type.unwrap_or_default())
        );
        crate::telemetry::telemetry_info!("embed_content started");

        let url = format!(
            "{}/models/{}:embedContent?key={}",
            self.api_url, request.model, self.api_key
        );

        let response = match self.http_client.post(&url).json(request).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = GeminiError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    "embed_content request failed"
                );
                return Err(error);
            }
        };

        if !response.status().is_success() {
            let error = GeminiError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::gemini_error_kind(&error),
                "embed_content API failure"
            );
            return Err(error);
        }

        let response: EmbedContentResponse = match response.json().await {
            Ok(response) => response,
            Err(error) => {
                let error = GeminiError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    "embed_content response parsing failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!("embed_content completed");

        Ok(response)
    }

    /// Generates embeddings for a batch of content in a single request.
    pub async fn batch_embed_contents(
        &self,
        model: &str,
        request: &BatchEmbedContentsRequest,
    ) -> Result<BatchEmbedContentsResponse, GeminiError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.batch_embed_contents",
            model,
            request_count = request.requests.len()
        );
        crate::telemetry::telemetry_info!("batch_embed_contents started");

        let url = format!(
            "{}/models/{}:batchEmbedContents?key={}",
            self.api_url, model, self.api_key
        );

        let response = match self.http_client.post(&url).json(request).send().await {
            Ok(response) => response,
            Err(error) => {
                let error = GeminiError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    "batch_embed_contents request failed"
                );
                return Err(error);
            }
        };

        if !response.status().is_success() {
            let error = GeminiError::from_response(response, None).await;
            crate::telemetry::telemetry_error!(
                error_kind = crate::telemetry::gemini_error_kind(&error),
                "batch_embed_contents API failure"
            );
            return Err(error);
        }

        let response: BatchEmbedContentsResponse = match response.json().await {
            Ok(response) => response,
            Err(error) => {
                let error = GeminiError::Http(error);
                crate::telemetry::telemetry_error!(
                    error_kind = crate::telemetry::gemini_error_kind(&error),
                    "batch_embed_contents response parsing failed"
                );
                return Err(error);
            }
        };

        crate::telemetry::telemetry_info!("batch_embed_contents completed");

        Ok(response)
    }

    /// Access the Files API client.
    pub fn files(&self) -> FilesClient<'_> {
        FilesClient { client: self }
    }
}

pub struct FilesClient<'a> {
    client: &'a GeminiClient,
}

impl<'a> FilesClient<'a> {
    /// Uploads a file to the Gemini File API.
    ///
    /// Automatically detects MIME type and selects the appropriate upload protocol
    /// (Multipart for small files, Resumable for large files).
    pub async fn upload_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<types::File, GeminiError> {
        let path = path.as_ref();
        let mime_type = mime_guess::from_path(path)
            .first_raw()
            .unwrap_or("application/octet-stream");
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");

        let metadata = std::fs::metadata(path).map_err(|e| {
            GeminiError::Api(serde_json::json!({
                "status": 500,
                "message": format!("Failed to read file metadata: {}", e),
            }))
        })?;
        let size = metadata.len();

        if size < 20 * 1024 * 1024 {
            self.upload_multipart(path, mime_type, file_name).await
        } else {
            self.upload_resumable(path, mime_type, file_name, size)
                .await
        }
    }

    async fn upload_multipart(
        &self,
        path: &std::path::Path,
        mime_type: &str,
        file_name: &str,
    ) -> Result<types::File, GeminiError> {
        let url = "https://generativelanguage.googleapis.com/upload/v1beta/files";
        let data = std::fs::read(path).map_err(|e| {
            GeminiError::Api(serde_json::json!({
                "status": 500,
                "message": format!("Failed to read file: {}", e),
            }))
        })?;

        let form = reqwest::multipart::Form::new()
            .part(
                "metadata",
                reqwest::multipart::Part::text(
                    serde_json::json!({
                        "file": { "display_name": file_name }
                    })
                    .to_string(),
                )
                .mime_str("application/json")?,
            )
            .part(
                "file",
                reqwest::multipart::Part::bytes(data).mime_str(mime_type)?,
            );

        let response = self
            .client
            .http_client
            .post(url)
            .query(&[("key", &self.client.api_key)])
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(GeminiError::from_response(response, None).await);
        }

        Ok(response.json().await?)
    }

    async fn upload_resumable(
        &self,
        path: &std::path::Path,
        mime_type: &str,
        file_name: &str,
        size: u64,
    ) -> Result<types::File, GeminiError> {
        let url = "https://generativelanguage.googleapis.com/upload/v1beta/files";

        // 1. Initial request to get upload URL
        let response = self
            .client
            .http_client
            .post(url)
            .query(&[("key", &self.client.api_key)])
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header("X-Goog-Upload-Header-Content-Length", size)
            .header("X-Goog-Upload-Header-Content-Type", mime_type)
            .json(&serde_json::json!({
                "file": { "display_name": file_name }
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(GeminiError::from_response(response, None).await);
        }

        let upload_url = response
            .headers()
            .get("X-Goog-Upload-URL")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                GeminiError::Api(serde_json::json!({"message": "Missing upload URL"}))
            })?;

        // 2. Upload the file content
        let file = tokio::fs::File::open(path).await.map_err(|e| {
            GeminiError::Api(serde_json::json!({
                "status": 500,
                "message": format!("Failed to open file for resumable upload: {}", e),
            }))
        })?;

        let response = self
            .client
            .http_client
            .post(upload_url)
            .header("X-Goog-Upload-Command", "upload, finalize")
            .header("X-Goog-Upload-Offset", 0)
            .body(file)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(GeminiError::from_response(response, None).await);
        }

        Ok(response.json().await?)
    }
}

/// Example:
/// ```rust
/// # use gemini_client_rs::gemini_role;
/// let role = gemini_role!(user);
/// ```

#[macro_export]
macro_rules! gemini_role {
    (user) => {
        $crate::types::Role::User
    };
    (model) => {
        $crate::types::Role::Model
    };
    ($role:ident) => {
        $crate::types::Role::$role
    };
}

/// A declarative macro to build a [GenerateContentRequest] quickly.
///
/// Example:
/// ```rust
/// # use gemini_client_rs::gemini_chat;
/// let req = gemini_chat!(
///     system("You are a helpful assistant"),
///     user("Hello!"),
///     model("Hi there! How can I help?"),
///     user("Tell me a joke.")
/// );
/// ```
#[macro_export]
macro_rules! gemini_chat {
    (system($sys:expr) $(, $role:ident($text:expr))*) => {
        $crate::types::GenerateContentRequest {
            system_instruction: Some($crate::types::Content {
                role: None,
                parts: vec![$crate::types::Part::text($sys)],
            }),
            contents: vec![
                $(
                    $crate::types::Content {
                        role: Some($crate::gemini_role!($role)),
                        parts: vec![$crate::types::Part::text($text)],
                    }
                ),*
            ],
            ..Default::default()
        }
    };
    ($($role:ident($text:expr)),*) => {
        $crate::types::GenerateContentRequest {
            contents: vec![
                $(
                    $crate::types::Content {
                        role: Some($crate::gemini_role!($role)),
                        parts: vec![$crate::types::Part::text($text)],
                    }
                ),*
            ],
            ..Default::default()
        }
    };
}

/// A declarative macro to build a list of [Part]s.
///
/// Example:
/// ```rust,no_run
/// # use gemini_client_rs::gemini_parts;
/// let parts = gemini_parts![
///     text("Analyze this image:"),
///     image("path/to/image.png")
/// ];
/// ```

#[macro_export]
macro_rules! gemini_parts {
    ($( $cmd:ident($arg:expr) ),* $(,)?) => {
        vec![
            $(
                $crate::gemini_part_internal!($cmd($arg))
            ),*
        ]
    };
}



#[doc(hidden)]
#[macro_export]
macro_rules! gemini_part_internal {
    (text($t:expr)) => {
        $crate::types::Part::text($t)
    };

    (image($p:expr)) => {{
        let path = std::path::Path::new($p);
        let data = std::fs::read(path).expect("Failed to read image file");
        let mime_type = $crate::get_mime_type(path);
        let base64_data = $crate::base64_encode(&data);
        $crate::types::Part::inline_data(mime_type, base64_data)
    }};

    (file_uri($u:expr)) => {
        $crate::types::Part::file_data("application/octet-stream", $u)
    };
    (thought($t:expr)) => {
        $crate::types::Part::thought($t)
    };
    (thought_signature($s:expr)) => {
        $crate::types::Part::ThoughtSignature { signature: $s.to_string() }
    };
}

#[doc(hidden)]
pub fn base64_encode(data: &[u8]) -> String {
    use base64::{engine::general_purpose, Engine as _};
    general_purpose::STANDARD.encode(data)
}

#[doc(hidden)]
pub fn get_mime_type(path: &std::path::Path) -> String {
    mime_guess::from_path(path)
        .first_raw()
        .unwrap_or("application/octet-stream")
        .to_string()
}

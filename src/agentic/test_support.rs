use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{
    agentic::{
        rag::{RagError, RagQuery, RetrievedChunk, Retriever},
        tool_runtime::ModelBackend,
    },
    types::{GenerateContentRequest, GenerateContentResponse},
    GeminiError,
};

pub(crate) type ScriptedResponse = Box<
    dyn Fn(&GenerateContentRequest) -> Result<GenerateContentResponse, GeminiError> + Send + Sync,
>;

pub(crate) struct ScriptedBackend {
    responses: Arc<Mutex<VecDeque<ScriptedResponse>>>,
}

impl ScriptedBackend {
    pub(crate) fn new(responses: Vec<ScriptedResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
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
            .expect("response queue lock")
            .pop_front()
            .ok_or_else(|| GeminiError::Api(json!({"error": "no scripted response remaining"})))?;

        response(request)
    }
}

pub(crate) struct StaticRetriever {
    chunks: Vec<RetrievedChunk>,
}

impl StaticRetriever {
    pub(crate) fn new(chunks: Vec<RetrievedChunk>) -> Self {
        Self { chunks }
    }
}

#[async_trait]
impl Retriever for StaticRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(self.chunks.iter().take(query.top_k).cloned().collect())
    }
}

pub(crate) fn response_with_text(text: &str) -> GenerateContentResponse {
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

pub(crate) fn response_with_function_calls(parts: Vec<Value>) -> GenerateContentResponse {
    serde_json::from_value(json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts
            }
        }],
        "usageMetadata": {}
    }))
    .expect("valid response payload")
}

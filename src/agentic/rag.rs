use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agentic::{extract_text_response, request_with_json_response},
    types::GenerateContentResponse,
    GeminiError,
};

use super::tool_runtime::ModelBackend;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RagQuery {
    pub question: String,
    pub top_k: usize,
    pub metadata_filter: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievedChunk {
    pub id: String,
    pub source: String,
    pub title: String,
    pub content: String,
    pub score: f64,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub top_k: usize,
    pub max_context_chars: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            max_context_chars: 12_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RagResponse {
    pub answer: String,
    pub cited_chunk_ids: Vec<String>,
    pub retrieved_chunks: Vec<RetrievedChunk>,
    pub raw_response: GenerateContentResponse,
}

#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error(transparent)]
    Backend(#[from] GeminiError),
    #[error("The retriever did not return any chunks for the question")]
    NoRetrievedChunks,
    #[error("The model did not return any text for {context}")]
    NoTextResponse { context: String },
    #[error("Failed to parse JSON for {context}: {error} (payload: {data})")]
    Json {
        context: String,
        data: String,
        #[source]
        error: serde_json::Error,
    },
    #[error("The model cited chunk ids that were not retrieved: {invalid_ids:?}")]
    InvalidCitationIds { invalid_ids: Vec<String> },
}

pub struct RagSession<'a, B, R> {
    backend: &'a B,
    retriever: &'a R,
    config: RagConfig,
}

impl<'a, B, R> RagSession<'a, B, R>
where
    B: ModelBackend,
    R: Retriever,
{
    pub fn new(backend: &'a B, retriever: &'a R, config: RagConfig) -> Self {
        Self {
            backend,
            retriever,
            config,
        }
    }

    pub async fn answer(
        &self,
        model: &str,
        question: &str,
        system_instruction: Option<&str>,
    ) -> Result<RagResponse, RagError> {
        let _span = crate::telemetry::telemetry_span_guard!(
            info,
            "gemini_client_rs.rag.answer",
            model = model,
            top_k = self.config.top_k,
            max_context_chars = self.config.max_context_chars,
            has_system_instruction = system_instruction.is_some()
        );
        crate::telemetry::telemetry_info!("rag.answer started");

        let query = RagQuery {
            question: question.to_string(),
            top_k: self.config.top_k,
            metadata_filter: None,
        };
        let retrieved_chunks = self.retriever.retrieve(&query).await?;
        crate::telemetry::telemetry_debug!(
            retrieval_count = retrieved_chunks.len(),
            "rag.answer retrieval completed"
        );

        if retrieved_chunks.is_empty() {
            crate::telemetry::telemetry_warn!("rag.answer returned no retrieved chunks");
            return Err(RagError::NoRetrievedChunks);
        }

        let retrieved_chunks = sort_and_limit_chunks(retrieved_chunks, self.config.top_k);
        let context_chunks =
            truncate_chunks_for_context(retrieved_chunks.clone(), self.config.max_context_chars);
        crate::telemetry::telemetry_debug!(
            selected_chunk_count = retrieved_chunks.len(),
            truncated_chunk_count = context_chunks.len(),
            "rag.answer prepared context"
        );

        let context = format_context(&context_chunks);
        let prompt = format!(
            "Answer the user's question using only the retrieved context.\nReturn JSON with keys \"answer\" and \"citation_chunk_ids\".\nEach citation must be one of the retrieved chunk ids.\n\nQuestion:\n{question}\n\nRetrieved context:\n{context}"
        );

        let schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" },
                "citation_chunk_ids": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["answer", "citation_chunk_ids"]
        });

        let request = request_with_json_response(system_instruction, prompt, schema);
        let raw_response = self.backend.generate_content(model, &request).await?;
        let payload_text =
            extract_text_response(&raw_response).ok_or_else(|| RagError::NoTextResponse {
                context: "rag answer".to_string(),
            })?;
        let payload =
            serde_json::from_str::<StructuredRagAnswer>(&payload_text).map_err(|error| {
                RagError::Json {
                    context: "rag answer".to_string(),
                    data: payload_text.clone(),
                    error,
                }
            })?;

        let valid_ids = context_chunks
            .iter()
            .map(|chunk| chunk.id.clone())
            .collect::<HashSet<_>>();
        let invalid_ids = payload
            .citation_chunk_ids
            .iter()
            .filter(|chunk_id| !valid_ids.contains(*chunk_id))
            .cloned()
            .collect::<Vec<_>>();

        if !invalid_ids.is_empty() {
            crate::telemetry::telemetry_warn!(
                invalid_citation_count = invalid_ids.len(),
                "rag.answer returned invalid citation ids"
            );
            return Err(RagError::InvalidCitationIds { invalid_ids });
        }

        crate::telemetry::telemetry_info!(
            citation_count = payload.citation_chunk_ids.len(),
            selected_chunk_count = context_chunks.len(),
            "rag.answer completed"
        );

        Ok(RagResponse {
            answer: payload.answer,
            cited_chunk_ids: payload.citation_chunk_ids,
            retrieved_chunks: context_chunks,
            raw_response,
        })
    }
}

#[derive(Debug, Deserialize)]
struct StructuredRagAnswer {
    answer: String,
    citation_chunk_ids: Vec<String>,
}

pub(crate) fn sort_and_limit_chunks(
    chunks: Vec<RetrievedChunk>,
    top_k: usize,
) -> Vec<RetrievedChunk> {
    let mut indexed = chunks.into_iter().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(left_index, left), (right_index, right)| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left_index.cmp(right_index))
    });

    indexed
        .into_iter()
        .take(top_k)
        .map(|(_, chunk)| chunk)
        .collect()
}

pub(crate) fn truncate_chunks_for_context(
    chunks: Vec<RetrievedChunk>,
    max_context_chars: usize,
) -> Vec<RetrievedChunk> {
    let mut used_chars = 0usize;
    let mut selected = vec![];

    for mut chunk in chunks {
        if used_chars >= max_context_chars {
            break;
        }

        let remaining = max_context_chars - used_chars;
        if chunk.content.chars().count() > remaining {
            chunk.content = chunk.content.chars().take(remaining).collect::<String>();
        }

        if chunk.content.is_empty() {
            continue;
        }

        used_chars += chunk.content.chars().count();
        selected.push(chunk);
    }

    selected
}

pub(crate) fn format_context(chunks: &[RetrievedChunk]) -> String {
    chunks
        .iter()
        .map(|chunk| {
            format!(
                "[chunk:{}]\nsource: {}\ntitle: {}\ncontent:\n{}",
                chunk.id, chunk.source, chunk.title, chunk.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::{
        format_context, sort_and_limit_chunks, truncate_chunks_for_context, RagConfig, RagError,
        RagSession, RetrievedChunk,
    };
    use crate::agentic::test_support::{response_with_text, ScriptedBackend, StaticRetriever};

    fn chunk(id: &str, score: f64, content: &str) -> RetrievedChunk {
        RetrievedChunk {
            id: id.to_string(),
            source: "kb".to_string(),
            title: format!("Doc {id}"),
            content: content.to_string(),
            score,
            metadata: None,
        }
    }

    #[test]
    fn retrieval_ordering_is_score_desc_then_input_order() {
        let ordered = sort_and_limit_chunks(
            vec![
                chunk("b", 0.8, "B"),
                chunk("a", 0.9, "A"),
                chunk("c", 0.8, "C"),
            ],
            3,
        );

        assert_eq!(
            ordered
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn context_truncation_is_deterministic() {
        let truncated = truncate_chunks_for_context(
            vec![chunk("a", 1.0, "12345"), chunk("b", 0.9, "67890")],
            7,
        );

        assert_eq!(truncated.len(), 2);
        assert_eq!(truncated[0].content, "12345");
        assert_eq!(truncated[1].content, "67");
    }

    #[test]
    fn context_format_is_stable() {
        let context = format_context(&[chunk("a", 1.0, "Alpha")]);
        assert!(context.contains("[chunk:a]"));
        assert!(context.contains("source: kb"));
        assert!(context.contains("content:\nAlpha"));
    }

    #[tokio::test]
    async fn validates_citation_ids_against_retrieved_chunks() {
        let backend = ScriptedBackend::new(vec![Box::new(|_| {
            Ok(response_with_text(
                r#"{"answer":"Done","citation_chunk_ids":["missing"]}"#,
            ))
        })]);
        let retriever = StaticRetriever::new(vec![chunk("a", 1.0, "Alpha")]);
        let session = RagSession::new(&backend, &retriever, RagConfig::default());

        let error = session
            .answer("gemini-test", "question", None)
            .await
            .expect_err("invalid citations should fail");

        assert!(matches!(
            error,
            RagError::InvalidCitationIds { invalid_ids } if invalid_ids == vec!["missing".to_string()]
        ));
    }

    #[tokio::test]
    async fn returns_empty_retrieval_errors() {
        let backend = ScriptedBackend::new(vec![]);
        let retriever = StaticRetriever::new(vec![]);
        let session = RagSession::new(&backend, &retriever, RagConfig::default());

        let error = session
            .answer("gemini-test", "question", None)
            .await
            .expect_err("empty retrieval should fail");

        assert!(matches!(error, RagError::NoRetrievedChunks));
    }

    #[tokio::test]
    async fn returns_valid_rag_answers_with_citations() {
        let backend = ScriptedBackend::new(vec![Box::new(|request| {
            assert_eq!(
                request
                    .generation_config
                    .as_ref()
                    .and_then(|config| config.candidate_count),
                Some(1)
            );
            Ok(response_with_text(
                r#"{"answer":"Done","citation_chunk_ids":["a"]}"#,
            ))
        })]);
        let retriever = StaticRetriever::new(vec![chunk("a", 1.0, "Alpha")]);
        let session = RagSession::new(&backend, &retriever, RagConfig::default());

        let response = session
            .answer("gemini-test", "question", Some("Use citations"))
            .await
            .expect("rag call should succeed");

        assert_eq!(response.answer, "Done");
        assert_eq!(response.cited_chunk_ids, vec!["a".to_string()]);
        assert_eq!(response.retrieved_chunks.len(), 1);
    }
}

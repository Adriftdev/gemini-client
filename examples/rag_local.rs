use async_trait::async_trait;
use dotenvy::dotenv;
use gemini_client_rs::{
    agentic::rag::{RagConfig, RagError, RagQuery, RagSession, RetrievedChunk, Retriever},
    GeminiClient,
};

struct LocalRetriever {
    chunks: Vec<RetrievedChunk>,
}

#[async_trait]
impl Retriever for LocalRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        let mut chunks = self.chunks.clone();
        chunks.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(chunks.into_iter().take(query.top_k).collect())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model_name =
        std::env::var("GEMINI_MODEL_NAME").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    let client = GeminiClient::new(api_key);

    let retriever = LocalRetriever {
        chunks: vec![
            RetrievedChunk {
                id: "doc-1".to_string(),
                source: "local".to_string(),
                title: "Support policy".to_string(),
                content: "Priority incidents must receive an acknowledgement within 15 minutes."
                    .to_string(),
                score: 0.98,
                metadata: None,
            },
            RetrievedChunk {
                id: "doc-2".to_string(),
                source: "local".to_string(),
                title: "Escalation handbook".to_string(),
                content: "Escalate payment outages directly to the on-call operations lead."
                    .to_string(),
                score: 0.82,
                metadata: None,
            },
        ],
    };

    let session = RagSession::new(&client, &retriever, RagConfig::default());
    let response = session
        .answer(
            &model_name,
            "How quickly do we acknowledge priority incidents?",
            Some("Answer using the retrieved context and cite the chunk ids you used."),
        )
        .await?;

    println!("Answer: {}", response.answer);
    println!("Citations: {:?}", response.cited_chunk_ids);

    Ok(())
}

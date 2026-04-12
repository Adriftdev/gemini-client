use std::collections::HashMap;

use async_trait::async_trait;
use dotenvy::dotenv;
use gemini_client_rs::{
    agentic::{
        multi_agent::{SupervisorConfig, SupervisorWorkflow},
        rag::{RagError, RagQuery, RetrievedChunk, Retriever},
        tool_runtime::{AgentTools, ToolRegistry},
    },
    types::{
        FunctionDeclaration, FunctionParameters, ParameterProperty, ParameterPropertyString, Tool,
        ToolConfigFunctionDeclaration,
    },
    FunctionHandler, GeminiClient,
};
use serde_json::json;

struct LocalRetriever {
    chunks: Vec<RetrievedChunk>,
}

#[async_trait]
impl Retriever for LocalRetriever {
    async fn retrieve(&self, query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(self.chunks.iter().take(query.top_k).cloned().collect())
    }
}

fn build_tools() -> AgentTools {
    let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
        function_declarations: vec![FunctionDeclaration {
            name: "get_incident_status".to_string(),
            description: "Returns the status of an incident.".to_string(),
            parameters: Some(FunctionParameters {
                parameter_type: "object".to_string(),
                properties: HashMap::from([(
                    "incident_id".to_string(),
                    ParameterProperty::String(ParameterPropertyString {
                        description: Some("Incident identifier".to_string()),
                        enum_values: None,
                    }),
                )]),
                required: Some(vec!["incident_id".to_string()]),
            }),
            parameters_json_schema: None,
            response: None,
        }],
    });

    let mut handlers = ToolRegistry::new();
    handlers.insert(
        "get_incident_status".to_string(),
        FunctionHandler::Sync(Box::new(|args| {
            let incident_id = args
                .get("incident_id")
                .and_then(|value| value.as_str())
                .unwrap_or("INC-000");
            Ok(json!({
                "incident_id": incident_id,
                "status": "investigating"
            }))
        })),
    );

    AgentTools::new(vec![tool], handlers)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");
    let model_name =
        std::env::var("GEMINI_MODEL_NAME").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    let client = GeminiClient::new(api_key);
    let tools = build_tools();
    let retriever = LocalRetriever {
        chunks: vec![RetrievedChunk {
            id: "doc-1".to_string(),
            source: "local".to_string(),
            title: "Incident playbook".to_string(),
            content: "Major incidents should include current status, customer impact, and next update time."
                .to_string(),
            score: 0.99,
            metadata: None,
        }],
    };
    let workflow = SupervisorWorkflow::new(&client, SupervisorConfig::default());

    let outcome = workflow
        .run(
            &model_name,
            "Prepare a short stakeholder update for incident INC-204.",
            Some(&tools),
            Some(&retriever),
        )
        .await?;

    println!("Final answer:\n{}", outcome.final_answer);

    Ok(())
}

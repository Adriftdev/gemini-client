use std::collections::HashMap;

use async_trait::async_trait;
use dotenvy::dotenv;
use gemini_client_rs::{
    agentic::{
        planning::{PlanningConfig, PlanningSession},
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

struct NoopRetriever;

#[async_trait]
impl Retriever for NoopRetriever {
    async fn retrieve(&self, _query: &RagQuery) -> Result<Vec<RetrievedChunk>, RagError> {
        Ok(vec![])
    }
}

fn build_tools() -> AgentTools {
    let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
        function_declarations: vec![FunctionDeclaration {
            name: "get_release_status".to_string(),
            description: "Returns the status of a release train.".to_string(),
            parameters: Some(FunctionParameters {
                parameter_type: "object".to_string(),
                properties: HashMap::from([(
                    "release".to_string(),
                    ParameterProperty::String(ParameterPropertyString {
                        description: Some("Release train name".to_string()),
                        enum_values: None,
                    }),
                )]),
                required: Some(vec!["release".to_string()]),
            }),
            parameters_json_schema: None,
            response: None,
        }],
    });

    let mut handlers = ToolRegistry::new();
    handlers.insert(
        "get_release_status".to_string(),
        FunctionHandler::Sync(Box::new(|args| {
            let release = args
                .get("release")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            Ok(json!({
                "release": release,
                "status": "green",
                "summary": format!("{release} is ready to ship")
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
    let session = PlanningSession::new(&client, PlanningConfig::default());

    let outcome = session
        .run(
            &model_name,
            "Check the release health for release-2026-04 and summarize whether it is ready to ship.",
            Some(&tools),
            Option::<&NoopRetriever>::None,
        )
        .await?;

    println!("Final answer:\n{}", outcome.final_answer);

    Ok(())
}

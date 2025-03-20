use std::collections::HashMap;

use gemini_client_rs::{
    types::{GenerateContentRequest, PartResponse},
    GeminiClient,
};

use dotenvy::dotenv;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-2.0-flash";

    let req_json = json!({
        "contents": [
            {
                "parts": [
                    {
                        "text": "Calculate the square root of 16"
                    }
                ],
                "role": "user"
            }
        ],
        "tools": [
            {
                "code_execution": {}
            }
        ]
    });

    let request = serde_json::from_value::<GenerateContentRequest>(req_json)?;
    let response = client
        .generate_content_with_function_calling(model_name, request, &HashMap::new())
        .await?;

    let candidates = response.candidates.unwrap();

    for candidate in &candidates {
        for part in &candidate.content.parts {
            match part {
                PartResponse::Text(text) => println!("{}", text),
                _ => { /* Ignore other part types as we are not using tools */ }
            }
        }
    }

    Ok(())
}

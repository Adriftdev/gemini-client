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
    let model_name = "gemini-2.0-flash-exp";

    let req_json = json!({
        "contents": [
            {
                "parts": [
                    {
                        "text": "What's the current weather in London, UK?"
                    }
                ],
                "role": "user"
            }
        ],
        "tools": [
             {
                "function_declarations": [
                    {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. 'San Francisco, CA'"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ]
            }
        ]
    });

    let request = serde_json::from_value::<GenerateContentRequest>(req_json)?;

    let mut function_handlers: HashMap<
        String,
        Box<dyn Fn(&mut serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
    > = HashMap::new();

    function_handlers.insert(
        "get_current_weather".to_string(),
        Box::new(|args: &mut serde_json::Value| {
            if let Some(_location) = args.get("location").and_then(|v| v.as_str()) {
                // This is a dummy implementation, would normally call an external API, etc.
                Ok(serde_json::json!({ "temperature": 15, "condition": "Cloudy" }))
            } else {
                Err("Missing 'location' argument".to_string())
            }
        }),
    );

    let response = client
        .generate_content_with_function_calling(model_name, request, &function_handlers)
        .await?;

    let candidates = response.candidates.unwrap();

    let first_candidate = candidates.first().unwrap();

    let first_part = first_candidate.content.parts.first().unwrap();

    let weather = match first_part {
        PartResponse::Text(text) => text,
        PartResponse::FunctionCall(_) => "Function call found",
        PartResponse::FunctionResponse(_) => "Function response found",
    };

    println!("{}", weather);

    Ok(())
}

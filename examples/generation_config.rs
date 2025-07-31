use dotenvy::dotenv;
use gemini_client_rs::{
    types::{ContentData, GenerateContentRequest},
    GeminiClient,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key.to_string());
    let model_name = "gemini-2.5-flash";

    let req_json = json!({
        "contents": [
            {
                "parts": [
                    {
                        "text": "Generate a happy greeting message"
                    }
                ],
                "role": "user"
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "A single emoji representing the mood of the message"
                    },
                    "message": {
                        "type": "string",
                        "description": "A greeting message"
                    }
                },
                "required": ["emoji", "message"]
            }
        }
    });

    let request: GenerateContentRequest = serde_json::from_value(req_json)?;

    let response = client.generate_content(model_name, &request).await?;

    for candidate in &response.candidates {
        for part in &candidate.content.parts {
            match &part.data {
                ContentData::Text(text) => println!("Response: {text}"),
                _ => println!("Unexpected response type"),
            }
        }
    }

    Ok(())
}

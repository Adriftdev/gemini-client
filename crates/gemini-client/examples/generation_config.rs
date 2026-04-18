use dotenvy::dotenv;
use gemini_client_rs::{
    gemini_chat,
    types::{GeminiSchema as _, GenerationConfig, Part},
    GeminiClient, GeminiSchema,
};

#[derive(GeminiSchema)]
#[allow(dead_code)]
struct GreetingResponse {
    /// A single emoji representing the mood of the message
    emoji: String,
    /// A greeting message
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-3-flash-preview";

    // Use gemini_chat! with structured output configuration
    let mut req = gemini_chat!(user("Generate a happy greeting message"));

    req.generation_config = Some(GenerationConfig {
        response_mime_type: Some("application/json".to_string()),
        response_schema: Some(GreetingResponse::schema()),
        ..Default::default()
    });

    let response = client.generate_content(model_name, &req).await?;

    for candidate in &response.candidates {
        if let Some(content_data) = &candidate.content {
            for part in &content_data.parts {
                if let Part::Text { text } = part {
                    println!("JSON Response: {}", text);
                }
            }
        }
    }

    Ok(())
}

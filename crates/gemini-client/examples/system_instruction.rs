use gemini_client_rs::{
    gemini_chat, 
    types::Part,
    GeminiClient,
};

use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-2.5-flash";

    // use gemini_chat! with the system(...) instruction block
    let req = gemini_chat!(
        system("You are Albert Einstein."),
        user("Who are you? What theories did you develop?")
    );

    let response = client.generate_content(model_name, &req).await?;

    for candidate in &response.candidates {
        if let Some(content_data) = &candidate.content {
            for part in &content_data.parts {
                if let Part::Text { text } = part {
                    println!("Text: {}", text);
                }
            }
        }
    }

    Ok(())
}


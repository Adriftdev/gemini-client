use gemini_client_rs::{gemini_chat, types::Part, GeminiClient};

use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-3-flash-preview";

    let req = gemini_chat!(
        system("You are a helpful assistant that provides weather information."),
        user("What's the weather like in London, UK?")
    );

    let response = client.generate_content(model_name, &req).await?;

    for candidate in &response.candidates {
        if let Some(content_data) = &candidate.content {
            for part in &content_data.parts {
                match &part {
                    Part::Text { text } => {
                        println!("{}", text);
                    }

                    _ => {
                        println!("Non-text part: {:?}", part);
                    }
                }
            }
        } else {
            println!("No content data");
        }
    }

    Ok(())
}

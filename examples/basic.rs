use gemini_client_rs::{
    types::{Content, ContentPart, GenerateContentRequest, PartResponse, Role},
    GeminiClient,
};

use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-1.5-flash"; // Or your desired model

    let request = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![ContentPart::Text(
                r#"
                What's the weather like in Belvoir, Grantham, UK? use celcius.     
                and is it safe for me to drive to work tomorrow, 
                which is located near market harbourer?
                Is there any flooding that could be an issue or heavy snow or icing?"#
                    .to_string(),
            )],
            role: Role::User,
        }],
        tools: None,
    };

    let response = client.generate_content(model_name, &request).await?;

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

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

    let poem_base64 = String::from("YSBtZW93IGhlcmUgYSBtZW93IHRoZXJlIGEgbWVvdyAuLi4=");

    // Use gemini_chat! combined with gemini_parts! for multi-part messages
    let mut req = gemini_chat!(
        user("finish the rest of this poem")
    );

    // Append the inline data part
    req.contents[0].parts.push(Part::inline_data("text/plain", poem_base64));

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


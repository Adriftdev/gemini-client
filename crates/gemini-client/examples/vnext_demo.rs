use gemini_client_rs::{GeminiClient, gemini_chat, gemini_parts, GeminiSchema, gemini_tool};
use gemini_client_rs::types::{GeminiTool, GeminiSchema as _};

#[derive(GeminiSchema)]
#[allow(dead_code)]
/// A request for a specific analysis
struct AnalysisRequest {
    /// The topic to analyze
    topic: String,
    /// Whether to include deep research
    deep_research: bool,
}

#[gemini_tool]
#[allow(dead_code)]
/// Gets the current weather for a location
fn get_weather(location: String) -> String {
    format!("The weather in {} is sunny.", location)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _client = GeminiClient::default();

    
    // 1. New Declarative Macro DX
    let _req = gemini_chat!(
        system("You are a helpful analyst."),
        user("Analyze the current economy.")
    );
    
    // 2. Multimodal construction via gemini_parts!
    let _parts = gemini_parts![
        text("Analyze this:"),
        file_uri("https://example.com/image.png")
    ];

    // 3. Structured Output via GeminiSchema
    let schema = AnalysisRequest::schema();
    println!("Schema: {:?}", schema);

    // 4. Tool Declaration via #[gemini_tool]
    // The struct is now named GetWeatherTool
    let tool_decl = GetWeatherTool::declaration();
    println!("Tool: {:?}", tool_decl);

    Ok(())
}

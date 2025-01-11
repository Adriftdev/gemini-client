use std::collections::HashMap;

use gemini_client::{
    types::{
        Content, ContentPart, DynamicRetrieval, DynamicRetrievalConfig, FunctionDeclaration,
        FunctionParameters, GenerateContentRequest, ParameterProperty, PartResponse, Role,
        ToolConfig, ToolConfigFunctionDeclaration,
    },
    GeminiClient,
};

use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-1.5-flash"; // Or your desired model

    let get_current_weather_fn = FunctionDeclaration {
        name: "get_current_weather".to_string(),
        description: "Get the current weather in a given location".to_string(),
        parameters: FunctionParameters {
            parameter_type: "OBJECT".to_string(),
            properties: {
                let mut props = HashMap::new();
                props.insert(
                    "location".to_string(),
                    ParameterProperty {
                        property_type: "string".to_string(),
                        description: "The city and state, e.g. 'San Francisco, CA'".to_string(),
                        enum_values: None,
                    },
                );
                props
            },
            required: Some(vec!["location".to_string()]),
        },
    };

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
        tools: Some(vec![ToolConfig::DynamicRetieval {
            google_search_retrieval: DynamicRetrieval {
                dynamic_retrieval_config: DynamicRetrievalConfig {
                    mode: "MODE_DYNAMIC".to_string(),
                    dynamic_threshold: 0.5,
                },
            },
        }]),
    };

    let mut function_handlers: HashMap<
        String,
        Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
    > = HashMap::new();
    function_handlers.insert(
        "get_current_weather".to_string(),
        Box::new(|args: serde_json::Value| {
            if let Some(_location) = args.get("location").and_then(|v| v.as_str()) {
                //println!("Calling get_current_weather with location: {}", location);
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
        _ => "No weather found",
    };

    println!("Weather: {}", weather);

    Ok(())
}

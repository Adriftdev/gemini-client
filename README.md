# gemini-client-rs

A Rust client for the Google Gemini API.

This crate provides a convenient way to interact with the Google Gemini API, allowing you to generate text, leverage function calling, and utilize grounding capabilities.

## Features

*   **Text Generation:** Generate text content using various Gemini models.
*   **Function Calling:** Define and call custom functions through the Gemini API.
*   **Grounding (with Google Search):** Integrate Google Search to enhance responses with real-time information.
*   **Easy-to-use API:** Provides a simple and intuitive interface for interacting with the Gemini API.
*   **Error Handling:** Includes a comprehensive error type (`GeminiError`) for handling API and other issues.

## Getting Started

### Prerequisites

*   A Google Cloud project with the Gemini API enabled.
*   An API key for the Gemini API.

### Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
gemini-client-rs = "0.1.0" # Replace with the actual version
tokio = { version = "1", features = ["full"] }
```

### Setting up your API Key

You need to obtain an API key from the Google Cloud console. You can then set the `GEMINI_API_KEY` environment variable or pass the key directly to the `GeminiClient::new()` constructor.

**Using `.env` file (recommended for development):**

1. Create a `.env` file in the root of your project.
2. Add your API key to the `.env` file:

    ```env
    GEMINI_API_KEY=YOUR_API_KEY
    ```

3. Load the `.env` file in your application:

    ```rust
    use dotenvy::dotenv;

    #[tokio::main]
    async fn main() -> Result<(), Box<dyn std::error::Error>> {
        dotenv().ok();
        // ... rest of your code
    }
    ```

## Examples

Here are some examples demonstrating how to use the `gemini-client-rs` crate.

### Basic Text Generation

```rust
use gemini_client::{
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
                "Write a short poem about the rust programming language.".to_string(),
            )],
            role: Role::User,
        }],
        tools: None,
    };

    let response = client.generate_content(model_name, &request).await?;

    if let Some(candidates) = response.candidates {
        for candidate in &candidates {
            for part in &candidate.content.parts {
                match part {
                    PartResponse::Text(text) => println!("Response: {}", text),
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
```

### Function Calling with a Custom Function

```rust
use std::collections::HashMap;
use gemini_client::{
    types::{
        Content, ContentPart, FunctionDeclaration, FunctionParameters, GenerateContentRequest,
        ParameterProperty, PartResponse, Role, ToolConfig, ToolConfigFunctionDeclaration,
    },
    GeminiClient,
};
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-2.0-flash-exp"; // Or your desired model

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
                "What's the current weather in London?".to_string(),
            )],
            role: Role::User,
        }],
        tools: Some(vec![ToolConfig::FunctionDeclaration(
            ToolConfigFunctionDeclaration {
                function_declarations: vec![get_current_weather_fn],
            },
        )]),
    };

    let mut function_handlers: HashMap<
        String,
        Box<dyn Fn(serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
    > = HashMap::new();

    function_handlers.insert(
        "get_current_weather".to_string(),
        Box::new(|args: serde_json::Value| {
            if let Some(location) = args.get("location").and_then(|v| v.as_str()) {
                println!("Fetching weather for: {}", location);
                // In a real application, you would make an external API call here.
                Ok(serde_json::json!({ "temperature": 10, "condition": "Rainy" }))
            } else {
                Err("Missing 'location' argument".to_string())
            }
        }),
    );

    let response = client
        .generate_content_with_function_calling(model_name, request, &function_handlers)
        .await?;

    if let Some(candidates) = response.candidates {
        if let Some(candidate) = candidates.first() {
            if let Some(part) = candidate.content.parts.first() {
                match part {
                    PartResponse::Text(text) => println!("Response: {}", text),
                    PartResponse::FunctionCall(fc) => println!("Function Call: {:?}", fc),
                    PartResponse::FunctionResponse(fr) => println!("Function Response: {:?}", fr),
                }
            }
        }
    }

    Ok(())
}
```

### Grounding with Google Search

```rust
use std::collections::HashMap;
use gemini_client::{
    types::{
        Content, ContentPart, DynamicRetrieval, DynamicRetrievalConfig, GenerateContentRequest,
        PartResponse, Role, ToolConfig,
    },
    GeminiClient,
};
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = GeminiClient::new(api_key);
    let model_name = "gemini-1.5-pro"; // Or your desired model

    let request = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![ContentPart::Text(
                "What are the current trending news topics?".to_string(),
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

    let response = client
        .generate_content_with_function_calling(model_name, request, &HashMap::new())
        .await?;

    if let Some(candidates) = response.candidates {
        for candidate in &candidates {
            for part in &candidate.content.parts {
                match part {
                    PartResponse::Text(text) => println!("Response: {}", text),
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
```

## Error Handling

The crate defines a `GeminiError` enum to represent various errors that can occur during API calls or function execution. You can handle these errors using standard Rust error handling mechanisms:

```rust
use gemini_client::{GeminiClient, types::GenerateContentRequest, GeminiError};

async fn generate(client: &GeminiClient, request: &GenerateContentRequest) -> Result<(), GeminiError> {
    let model_name = "gemini-1.5-flash";
    match client.generate_content(model_name, request).await {
        Ok(response) => {
            println!("Response received: {:?}", response);
            Ok(())
        }
        Err(e) => {
            eprintln!("Error generating content: {}", e);
            Err(e)
        }
    }
}
```

## Supported Models

The examples use `gemini-1.5-flash`, `gemini-1.5-pro`, and `gemini-2.0-flash-exp`. Refer to the Google Gemini API documentation for the latest list of available models.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs or feature requests.

## License

This project is licensed under the MIT License.

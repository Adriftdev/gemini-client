# gemini_client_rs

`gemini_client_rs` is a transport-focused Rust SDK for the Google Gemini API.

It provides:

- typed request and response models
- content generation
- streaming content generation
- model listing
- lightweight telemetry hooks

This crate does not own orchestration, planning, retrieval, or tool-loop execution. Those behaviors should live in the application layer, such as RAIN.

## Basic usage

```rust
use gemini_client_rs::{gemini_chat, GeminiClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = GeminiClient::default();
    
    // Use the declarative gemini_chat! macro for elegant request building
    let req = gemini_chat!(
        user("Summarize this project in two sentences.")
    );

    let response = client
        .generate_content("gemini-2.0-flash", &req)
        .await?;

    println!("{response:#?}");
    Ok(())
}
```


## Position in the stack

- Use `gemini_client_rs` when you want a low-level SDK for Gemini.
- Use RAIN when you want agentic execution, tool orchestration, retrieval, planning, or multi-step workflows.

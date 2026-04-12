# gemini-client-rs

A Rust client for the Google Gemini API with a higher-level `agentic` layer for tool use, app-owned RAG, bounded planning, and deterministic supervisor workflows.

## Features

- Low-level Gemini API client via `GeminiClient`
- Native Gemini tools and grounding support through `types::Tool`
- Reusable tool runtime for function-calling loops
- App-owned RAG with retriever traits and citation validation
- Bounded plan/execute/evaluate orchestration
- Deterministic supervisor/worker/reviewer/synthesizer workflow

## Installation

```toml
[dependencies]
gemini_client_rs = "0.7.0"
tokio = { version = "1", features = ["full"] }
dotenvy = "0.15"
async-trait = "0.1"

[dependencies.tracing-subscriber]
version = "0.3"
features = ["fmt"]
```

## API Key

Set `GEMINI_API_KEY` in your environment or a local `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

## Low-Level Usage

```rust
use dotenvy::dotenv;
use gemini_client_rs::{
    types::{ContentData, GenerateContentRequest},
    GeminiClient,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let client = GeminiClient::new(std::env::var("GEMINI_API_KEY")?);
    let request = serde_json::from_value::<GenerateContentRequest>(json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": "Explain the purpose of a release checklist." }]
        }]
    }))?;

    let response = client.generate_content("gemini-2.5-flash", &request).await?;

    for candidate in &response.candidates {
        if let Some(content) = &candidate.content {
            for part in &content.parts {
                if let ContentData::Text(text) = &part.data {
                    println!("{text}");
                }
            }
        }
    }

    Ok(())
}
```

## Function Calling

The legacy `GeminiClient::generate_content_with_function_calling` API is still available. Internally it now delegates to the reusable `agentic::tool_runtime` loop, which:

- reads candidate `0` only for deterministic orchestration,
- processes all function-call parts in order,
- appends one function response per executed call,
- stops when the model stops requesting tools,
- errors if the loop exceeds `max_round_trips`.

For higher-level orchestration, construct `AgentTools` with both tool declarations and handlers:

```rust
use std::collections::HashMap;

use gemini_client_rs::{
    agentic::tool_runtime::{AgentTools, ToolRegistry},
    types::{
        FunctionDeclaration, FunctionParameters, ParameterProperty, ParameterPropertyString, Tool,
        ToolConfigFunctionDeclaration,
    },
    FunctionHandler,
};
use serde_json::json;

let tool = Tool::FunctionDeclaration(ToolConfigFunctionDeclaration {
    function_declarations: vec![FunctionDeclaration {
        name: "lookup_status".to_string(),
        description: "Looks up the status of a service".to_string(),
        parameters: Some(FunctionParameters {
            parameter_type: "object".to_string(),
            properties: HashMap::from([(
                "service".to_string(),
                ParameterProperty::String(ParameterPropertyString {
                    description: Some("Service name".to_string()),
                    enum_values: None,
                }),
            )]),
            required: Some(vec!["service".to_string()]),
        }),
        parameters_json_schema: None,
        response: None,
    }],
});

let mut handlers = ToolRegistry::new();
handlers.insert(
    "lookup_status".to_string(),
    FunctionHandler::Sync(Box::new(|args| {
        Ok(json!({ "status": format!("{} is healthy", args["service"]) }))
    })),
);

let agent_tools = AgentTools::new(vec![tool], handlers);
```

## Agentic Layer

The `agentic` module adds three high-level patterns on top of the low-level client.

### RAG

- `rag::Retriever` lets your application own retrieval.
- `rag::RagSession` retrieves chunks, builds deterministic context, asks Gemini for structured JSON, and validates citation ids.
- Gemini-native search/grounding remains separate and complementary.

See [examples/rag_local.rs](/Users/adrift/projects/gemini_client/examples/rag_local.rs).

### Planning

- `planning::PlanningSession` runs a bounded planner/executor/evaluator loop.
- Step schema is fixed: `id`, `title`, `instruction`, `success_criteria`, `allowed_tools`, `needs_rag`.
- Each run keeps in-memory working memory only.
- Planner and evaluator turns force `candidate_count = 1` and `response_mime_type = application/json`.

See [examples/plan_and_execute.rs](/Users/adrift/projects/gemini_client/examples/plan_and_execute.rs).

### Multi-Agent

- `multi_agent::SupervisorWorkflow` implements one deterministic topology only:
  supervisor -> worker -> reviewer -> synthesizer
- Work happens sequentially.
- Reviewer-triggered revisions are capped at one rerun per artifact.
- All coordination goes through an append-only `Blackboard`.

See [examples/supervisor_workflow.rs](/Users/adrift/projects/gemini_client/examples/supervisor_workflow.rs).

## Tracing

Tracing support is available behind the optional `tracing` feature:

```toml
[dependencies]
gemini_client_rs = { version = "0.7.0", features = ["tracing"] }
tracing-subscriber = { version = "0.3", features = ["fmt"] }
```

The library does not install a subscriber for you. Configure one in your application:

```rust
use tracing_subscriber::EnvFilter;

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("gemini_client_rs=info")),
        )
        .try_init();
}
```

The emitted logs are lifecycle-only. They include model names, counts, retry/replan decisions, tool names, and error kinds, but they do not include prompts, model output bodies, tool arguments, or retrieved document contents.

## Built-In Examples

```bash
cargo run --example basic
cargo run --example custom_tool
cargo run --example rag_local
cargo run --example plan_and_execute
cargo run --example supervisor_workflow
cargo test --features tracing
```

## Verification

The crate is verified with:

```bash
cargo test
cargo test --features tracing
cargo check --examples
cargo check --features tracing --examples
```

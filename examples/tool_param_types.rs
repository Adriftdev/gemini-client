use std::collections::HashMap;

/// Example demonstrating how to build tool schemas using Rust types for the Gemini API.
///
/// This example shows:
/// 1. How to construct FunctionDeclaration with complex parameter types (arrays, strings, integers, booleans)
/// 2. How to handle function calls with proper parameter validation using serde structs
/// 3. How to provide system instructions with current date context
///
/// The example builds a "schedule_meeting" tool that accepts:
/// - attendees: array of strings
/// - date: string (ISO date format)
/// - time: string (time format)
/// - topic: string (meeting subject)
/// - priority: integer (priority level from 1 to 10)
/// - category: string (meeting category: personal, work, or family)
/// - is_public: boolean (whether others can see meeting details, defaults to true)
///
/// ## Usage
///
/// ### Schema Verification Only (no API key required):
/// ```bash
/// cargo run --example tool_param_types
/// ```
/// This will build the tool schema, serialize it to JSON, and verify it matches
/// the expected structure.
///
/// ### With API Call (requires GEMINI_API_KEY):
/// ```bash
/// export GEMINI_API_KEY="your-api-key-here"
/// cargo run --example tool_param_types
/// ```
/// This will additionally make an actual API call to test the tool functionality.
///
/// ### With Custom Model (optional GEMINI_MODEL_NAME):
/// ```bash
/// export GEMINI_API_KEY="your-api-key-here"
/// export GEMINI_MODEL_NAME="gemini-1.5-pro"
/// cargo run --example tool_param_types
/// ```
/// This will use the specified model instead of the default "gemini-2.5-flash".
use gemini_client_rs::{
    types::{
        Content, ContentData, ContentPart, FunctionDeclaration, FunctionParameters,
        GenerateContentRequest, ParameterProperty, ParameterPropertyArray,
        ParameterPropertyBoolean, ParameterPropertyInteger, ParameterPropertyString, Role, Tool,
        ToolConfigFunctionDeclaration,
    },
    GeminiClient,
};

use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct MeetingRequest {
    attendees: Vec<String>,
    date: String,
    time: String,
    topic: String,
    priority: i64,
    category: String,
    is_public: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct MeetingResponse {
    status: String,
    meeting_id: String,
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let current_date = "2025-06-19";
    let system_message = format!(
        "You are a helpful assistant. Today's date is {}. When scheduling meetings, use appropriate dates relative to today.",
        current_date
    );

    let properties = HashMap::from([
        // attendees: array of strings
        (
            "attendees".to_string(),
            ParameterProperty::Array(ParameterPropertyArray {
                description: Some("List of people attending the meeting.".to_string()),
                items: Box::new(ParameterProperty::String(ParameterPropertyString {
                    description: None,
                    enum_values: None,
                })),
            }),
        ),
        // date: string
        (
            "date".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("Date of the meeting (e.g., '2024-07-29')".to_string()),
                enum_values: None,
            }),
        ),
        // time: string
        (
            "time".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("Time of the meeting (e.g., '15:00')".to_string()),
                enum_values: None,
            }),
        ),
        // topic: string
        (
            "topic".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("The subject or topic of the meeting.".to_string()),
                enum_values: None,
            }),
        ),
        // priority: integer
        (
            "priority".to_string(),
            ParameterProperty::Integer(ParameterPropertyInteger {
                description: Some("Priority level of the meeting from 1 to 10".to_string()),
            }),
        ),
        // category: string with enum values
        (
            "category".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("Category of the meeting".to_string()),
                enum_values: Some(vec![
                    "personal".to_string(),
                    "work".to_string(),
                    "family".to_string(),
                ]),
            }),
        ),
        // is_public: boolean
        (
            "is_public".to_string(),
            ParameterProperty::Boolean(ParameterPropertyBoolean {
                description: Some(
                    "Whether others can see meeting details (defaults to true)".to_string(),
                ),
            }),
        ),
    ]);

    let response_properties = HashMap::from([
        // No response properties defined for this function
        // This can be extended later if needed
        (
            "status".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("Status of the meeting scheduling operation.".to_string()),
                enum_values: None,
            }),
        ),
        (
            "meeting_id".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("Unique identifier for the scheduled meeting.".to_string()),
                enum_values: None,
            }),
        ),
        (
            "message".to_string(),
            ParameterProperty::String(ParameterPropertyString {
                description: Some("Detailed message about the scheduling result.".to_string()),
                enum_values: None,
            }),
        ),
    ]);

    let function_declaration = FunctionDeclaration {
        name: "schedule_meeting".to_string(),
        description: "Schedules a meeting with specified attendees at a given time and date."
            .to_string(),
        parameters: Some(FunctionParameters {
            parameter_type: "object".to_string(),
            properties,
            required: Some(vec![
                "attendees".to_string(),
                "date".to_string(),
                "time".to_string(),
                "topic".to_string(),
                "priority".to_string(),
                "category".to_string(),
                "is_public".to_string(),
            ]),
        }),
        response: Some(FunctionParameters {
            parameter_type: "object".to_string(),
            properties: response_properties,
            required: Some(vec![
                "status".to_string(),
                "meeting_id".to_string(),
                "message".to_string(),
            ]),
        }),
    };

    let request = GenerateContentRequest {
        system_instruction: Some(Content {
            parts: vec![ContentPart {
                data: ContentData::Text(system_message),
                thought: false,
                metadata: None,
            }],
            role: Role::User,
        }),
        contents: vec![Content {
            parts: vec![ContentPart{
                data: ContentData::Text( "Please schedule a team meeting for tomorrow at 2 PM with John, Sarah, and Mike to discuss the quarterly review. tag it as work, it's a P5, and make it public".to_string()),
                metadata: None,
                thought: false
            }],
            role: Role::User,
        }],
        tools: vec![
            Tool::FunctionDeclaration(
                ToolConfigFunctionDeclaration{
                    function_declarations: vec![
                        function_declaration.clone()
                                    ]
                }
            ),
        ],
        tool_config: None,
        generation_config: None,
    };

    // Expected JSON schema for comparison
    let expected_schema = json!({
        "name": "schedule_meeting",
        "description": "Schedules a meeting with specified attendees at a given time and date.",
        "parameters": {
            "type": "object",
            "properties": {
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of people attending the meeting.",
                },
                "date": {
                    "type": "string",
                    "description": "Date of the meeting (e.g., '2024-07-29')",
                },
                "time": {
                    "type": "string",
                    "description": "Time of the meeting (e.g., '15:00')",
                },
                "topic": {
                    "type": "string",
                    "description": "The subject or topic of the meeting.",
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority level of the meeting from 1 to 10",
                },
                "category": {
                    "type": "string",
                    "description": "Category of the meeting",
                    "enum": ["personal", "work", "family"],
                },
                "is_public": {
                    "type": "boolean",
                    "description": "Whether others can see meeting details (defaults to true)",
                },
            },
            "required": ["attendees", "date", "time", "topic", "priority", "category", "is_public"],
        },
        "response": {
            "properties": {
              "meeting_id": {
                "description": "Unique identifier for the scheduled meeting.",
                "type": "string"
              },
              "message": {
                "description": "Detailed message about the scheduling result.",
                "type": "string"
              },
              "status": {
                "description": "Status of the meeting scheduling operation.",
                "type": "string"
              }
            },
            "required": [
              "status",
              "meeting_id",
              "message"
            ],
            "type": "object"
        },
    });

    // Serialize the function declaration to JSON for verification
    let serialized_function = serde_json::to_value(&function_declaration)?;

    println!("Serialized function declaration:");
    println!("{}", serde_json::to_string_pretty(&serialized_function)?);

    println!("\nExpected schema:");
    println!("{}", serde_json::to_string_pretty(&expected_schema)?);

    // Verify the schemas match by comparing their string representations
    let serialized_str = serde_json::to_string(&serialized_function)?;
    let expected_str = serde_json::to_string(&expected_schema)?;

    assert_eq!(
        serialized_str, expected_str,
        "Schema mismatch!\nSerialized: {}\nExpected: {}",
        serialized_str, expected_str
    );

    println!("\n✅ Schema verification passed!");
    println!("📋 The Rust types successfully generate a valid tool schema for the Gemini API!");

    // Example of how to use with API (requires GEMINI_API_KEY):
    if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
        println!("\n🚀 Making API call...");

        let client = GeminiClient::new(api_key);
        let model_name =
            std::env::var("GEMINI_MODEL_NAME").unwrap_or_else(|_| "gemini-2.5-flash".to_string());

        // Set up function handler
        let mut function_handlers: HashMap<
            String,
            Box<dyn Fn(&mut serde_json::Value) -> Result<serde_json::Value, String> + Send + Sync>,
        > = HashMap::new();

        function_handlers.insert(
            "schedule_meeting".to_string(),
            Box::new(|args: &mut serde_json::Value| {
                // Deserialize the arguments using the MeetingRequest struct
                let meeting_request: MeetingRequest = serde_json::from_value(args.clone())
                    .map_err(|e| format!("Failed to deserialize meeting request: {}", e))?;

                // Print the meeting schedule details
                println!("📅 Meeting Details:");
                println!("  Topic: {}", meeting_request.topic);
                println!("  Date: {}", meeting_request.date);
                println!("  Time: {}", meeting_request.time);
                println!("  Attendees: {}", meeting_request.attendees.join(", "));
                println!("  Priority: {}", meeting_request.priority);
                println!("  Category: {}", meeting_request.category);
                println!("  Public: {}", meeting_request.is_public);

                // Create the response using the MeetingResponse struct
                let response = MeetingResponse {
                    status: "scheduled".to_string(),
                    meeting_id: "MTG-2024-001".to_string(),
                    message: format!(
                        "Meeting '{}' scheduled for {} at {} with {} attendees: {} (Priority: {}, Category: {}, Public: {})",
                        meeting_request.topic,
                        meeting_request.date,
                        meeting_request.time,
                        meeting_request.attendees.len(),
                        meeting_request.attendees.join(", "),
                        meeting_request.priority,
                        meeting_request.category,
                        meeting_request.is_public
                    ),
                };

                // Serialize the response back to JSON
                serde_json::to_value(response)
                    .map_err(|e| format!("Failed to serialize meeting response: {}", e))
            }),
        );

        // Make the request
        match client
            .generate_content_with_function_calling(&model_name, request, &function_handlers)
            .await
        {
            Ok(response) => {
                let candidates = response.candidates.iter().collect::<Vec<_>>();
                let first_candidate = candidates.first().unwrap();
                let first_part = first_candidate.content.parts.first().unwrap();

                let result = match first_part {
                    ContentPart {
                        data: ContentData::Text(text),
                        thought: false,
                        metadata: None,
                    } => text.clone(),
                    ContentPart {
                        data: ContentData::FunctionResponse(result),
                        thought: false,
                        metadata: None,
                    } => result.response.content.to_string(),
                    _ => "No valid response found".to_string(),
                };

                println!("🤖 Model Response:");
                println!("{}", result);
            }
            Err(e) => {
                println!("❌ API call failed: {}", e);
            }
        }
    } else {
        println!("\n💡 Set GEMINI_API_KEY environment variable to test the actual API call.");
    }

    Ok(())
}

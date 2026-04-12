use serde_json::Value;

use crate::types::{
    Content, ContentData, ContentPart, GenerateContentRequest, GenerateContentResponse,
};

pub mod multi_agent;
pub mod planning;
pub mod rag;
pub mod tool_runtime;

#[cfg(test)]
pub(crate) mod test_support;

pub(crate) fn build_system_instruction(text: Option<&str>) -> Option<Content> {
    text.map(|value| Content {
        parts: vec![ContentPart::new_text(value, false)],
        role: None,
    })
}

pub(crate) fn build_user_content(text: &str) -> Content {
    Content {
        parts: vec![ContentPart::new_text(text, false)],
        role: Some(crate::types::Role::User),
    }
}

pub(crate) fn extract_text_response(response: &GenerateContentResponse) -> Option<String> {
    response
        .candidates
        .first()
        .and_then(|candidate| candidate.content.as_ref())
        .map(|content| {
            content
                .parts
                .iter()
                .filter_map(|part| match &part.data {
                    ContentData::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty())
}

pub(crate) fn request_with_json_response(
    system_instruction: Option<&str>,
    user_prompt: String,
    schema: Value,
) -> GenerateContentRequest {
    GenerateContentRequest {
        system_instruction: build_system_instruction(system_instruction),
        contents: vec![build_user_content(&user_prompt)],
        tools: vec![],
        tool_config: None,
        generation_config: Some(crate::types::GenerationConfig {
            candidate_count: Some(1),
            response_mime_type: Some("application/json".to_string()),
            response_schema: Some(schema),
            ..Default::default()
        }),
    }
}

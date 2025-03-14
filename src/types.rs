use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "system")]
    System,
    #[serde(rename = "model")]
    Model,
    #[serde(rename = "tool")]
    Tool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateContentRequest {
    pub system_instruction: Option<Content>,
    pub contents: Vec<Content>,
    pub tools: Option<Vec<ToolConfig>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolConfig {
    #[serde(rename = "function_declaration")]
    FunctionDeclaration(ToolConfigFunctionDeclaration),
    DynamicRetieval {
        google_search_retrieval: DynamicRetrieval,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Content {
    pub parts: Vec<ContentPart>,
    pub role: Role,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "functionCall")]
    FunctionCall(FunctionCall),
    #[serde(rename = "functionResponse")]
    FunctionResponse(FunctionResponse),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolConfigFunctionDeclaration {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename = "google_search_retrieval")]
pub struct DynamicRetrieval {
    pub dynamic_retrieval_config: DynamicRetrievalConfig,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename = "dynamic_retrieval_config")]
pub struct DynamicRetrievalConfig {
    pub mode: String,
    pub dynamic_threshold: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: FunctionParameters,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub parameter_type: String,
    pub properties: HashMap<String, ParameterProperty>,
    pub required: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ParameterProperty {
    #[serde(rename = "type")]
    pub property_type: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Option<Vec<Candidate>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub content: ContentResponse,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContentResponse {
    pub parts: Vec<PartResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum PartResponse {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "functionCall")]
    FunctionCall(FunctionCall),
    #[serde(rename = "functionResponse")]
    FunctionResponse(FunctionResponse),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    #[serde(rename = "args")]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: FunctionResponsePayload,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionResponsePayload {
    pub content: serde_json::Value,
}

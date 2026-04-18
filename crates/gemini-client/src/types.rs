use std::collections::{BTreeSet, HashMap};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Schema {
    #[serde(rename = "type")]
    pub schema_type: SchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Schema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Schema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SchemaType {
    #[default]
    TypeUnspecified,
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}


#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[default]
    User,
    Model,
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contents: Vec<Content>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged, rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum Tool {


    // will work for both v1 and v2 models
    #[serde(rename = "function_declaration")]
    FunctionDeclaration(ToolConfigFunctionDeclaration),

    /* NOTE: For v1 models will be depreciated by google in 2025 */
    DynamicRetrieval {
        google_search_retrieval: DynamicRetrieval,
    },

    /* NOTE: Used by v2 models if they have search built in */
    GoogleSearch {
        google_search: serde_json::Value,
    },

    UrlContext {
        url_context: serde_json::Value,
    },

    /* NOTE: Used by v2 models if they have the code execution built in */
    CodeExecution {
        code_execution: serde_json::Value,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {

    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {

    pub mode: FunctionCallingMode,
    /// A set of function names that, when provided, limits the functions the
    /// model will call.
    ///
    /// This should only be set when the Mode is ANY. Function names should match
    /// [FunctionDeclaration.name]. With mode set to ANY, model will predict a
    /// function call from the set of function names provided.
    #[serde(default)]
    pub allowed_function_names: Vec<String>,
}

/// Defines the execution behavior for function calling by defining the execution
/// mode.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    /// Unspecified function calling mode. This value should not be used.
    #[default]
    ModeUnspecified,
    /// Default model behavior, model decides to predict either a function call
    /// or a natural language response.
    Auto,
    /// Model is constrained to always predicting a function call only. If
    /// "allowedFunctionNames" are set, the predicted function call will be
    /// limited to any one of "allowedFunctionNames", else the predicted
    /// function call will be any one of the provided "functionDeclarations".
    Any,
    /// Model will not predict any function call. Model behavior is same as when
    /// not passing any function declarations.
    None,
    /// Model decides to predict either a function call or a natural language
    /// response, but will validate function calls with constrained decoding.
    Validated,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Content {


    #[serde(default)]
    pub parts: Vec<Part>,
    // Optional. The producer of the content. Must be either 'user' or 'model'.
    // Useful to set for multi-turn conversations, otherwise can be left blank or unset.
    pub role: Option<Role>,
}


pub trait GeminiSchema {
    fn schema() -> Schema;
}

pub trait GeminiTool {
    fn declaration() -> FunctionDeclaration;
}

impl GeminiSchema for String {

    fn schema() -> Schema {
        Schema {
            schema_type: SchemaType::String,
            ..Default::default()
        }
    }
}

impl GeminiSchema for i32 {
    fn schema() -> Schema {
        Schema {
            schema_type: SchemaType::Integer,
            ..Default::default()
        }
    }
}

impl GeminiSchema for f64 {
    fn schema() -> Schema {
        Schema {
            schema_type: SchemaType::Number,
            ..Default::default()
        }
    }
}

impl GeminiSchema for bool {
    fn schema() -> Schema {
        Schema {
            schema_type: SchemaType::Boolean,
            ..Default::default()
        }
    }
}

impl<T: GeminiSchema> GeminiSchema for Vec<T> {
    fn schema() -> Schema {
        Schema {
            schema_type: SchemaType::Array,
            items: Some(Box::new(T::schema())),
            ..Default::default()
        }
    }
}

impl<T: GeminiSchema> GeminiSchema for Option<T> {
    fn schema() -> Schema {
        let mut s = T::schema();
        s.nullable = Some(true);
        s
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Schema>,

    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "_responseJsonSchema"
    )]
    pub response_json_schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub response_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_enhanced_civic_answers: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speech_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfigFunctionDeclaration {

    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRetrieval {
    pub dynamic_retrieval_config: DynamicRetrievalConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct DynamicRetrievalConfig {
    pub mode: String,
    pub dynamic_threshold: f64,
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct FunctionDeclaration {

    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Schema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Schema>,
}


/// [DEPRECATED] Use [Schema] instead.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(deprecated)]
#[deprecated(since = "0.10.0", note = "Use Schema instead")]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub parameter_type: String,
    #[allow(deprecated)]
    pub properties: HashMap<String, ParameterProperty>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

/// [DEPRECATED] Use [Schema] instead.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
#[allow(deprecated)]
#[deprecated(since = "0.10.0", note = "Use Schema instead")]
pub enum ParameterProperty {


    String(ParameterPropertyString),
    Integer(ParameterPropertyInteger),
    Number(ParameterPropertyNumber),
    Boolean(ParameterPropertyBoolean),
    Array(ParameterPropertyArray),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ParameterPropertyArray {

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[allow(deprecated)]
    pub items: Box<ParameterProperty>,

}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ParameterPropertyString {

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_values: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ParameterPropertyInteger {

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ParameterPropertyNumber {

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ParameterPropertyBoolean {

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Response from the model supporting multiple candidate responses.
///
/// Safety ratings and content filtering are reported for both prompt in
/// GenerateContentResponse.prompt_feedback and for each candidate in
/// finishReason and in safetyRatings.
///
/// The API:
/// - Returns either all requested candidates or none of them
/// - Returns no candidates at all only if there was something wrong with the
///   prompt (check promptFeedback)
/// - Reports feedback on each candidate in finishReason and safetyRatings.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {

    #[serde(default)]
    pub candidates: Vec<Candidate>,
    pub prompt_feedback: Option<PromptFeedback>,
    pub usage_metadata: UsageMetadata,
    #[serde(default)]
    pub model_version: Option<String>,
    #[serde(default)]
    pub response_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentRequest {

    pub model: String,
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<TaskType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskType {
    #[default]
    TaskTypeUnspecified,
    RetrievalQuery,
    RetrievalDocument,
    SemanticSimilarity,
    Classification,
    Clustering,
    QuestionAnswering,
    FactVerification,
    #[serde(other)]
    Other,
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct EmbedContentResponse {

    pub embedding: ContentEmbedding,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ContentEmbedding {

    pub values: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct BatchEmbedContentsRequest {

    pub requests: Vec<EmbedContentRequest>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct BatchEmbedContentsResponse {

    pub embeddings: Vec<ContentEmbedding>,
}

/// Specifies the reason why the prompt was blocked.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PromptFeedback {
    /// Default value. This value is unused.
    #[default]
    BlockReasonUnspecified,
    /// Prompt was blocked due to safety reasons. Inspect safetyRatings to
    /// understand which safety category blocked it.
    Safety,
    /// Prompt was blocked due to unknown reasons.
    Other,
    /// Prompt was blocked due to the terms which are included from the
    /// terminology blocklist.
    Blocklist,
    /// Prompt was blocked due to prohibited content.
    ProhibitedContent,
    /// Candidates blocked due to unsafe image generation content.
    ImageSafety,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_prompt_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<u32>,
    #[serde(default)]
    pub prompt_tokens_details: Vec<ModalityTokenCount>,
    #[serde(default)]
    pub cache_tokens_details: Vec<ModalityTokenCount>,
    #[serde(default)]
    pub candidates_tokens_details: Vec<ModalityTokenCount>,
    #[serde(default)]
    pub tool_use_prompt_tokens_details: Vec<ModalityTokenCount>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traffic_type: Option<TrafficType>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModalityTokenCount {
    pub modality: Modality,
    pub token_count: u32,
}


/// Request traffic type. Indicates whether the request consumes Pay-As-You-Go or
/// Provisioned Throughput quota.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TrafficType {
    /// Unspecified request traffic type.
    #[default]
    TrafficTypeUnspecified,
    /// Type for Pay-As-You-Go traffic.
    OnDemand,
    /// Type for Provisioned Throughput traffic.
    ProvisionedThroughput,
    #[serde(other)]
    Other,
}


/// Content Part modality
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Modality {
    /// Unspecified modality.
    #[default]
    ModalityUnspecified,
    /// Plain text.
    Text,
    /// Image.
    Image,
    /// Video.
    Video,
    /// Audio.
    Audio,
    /// Document, e.g. PDF.
    Document,
    /// Thinking / Reasoning.
    Thoughts,
    #[serde(other)]
    Other,
}


/// Config for thinking features.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {

    /// Indicates whether to include thoughts in the response. If true, thoughts
    /// are returned only when available.
    pub include_thoughts: bool,
    /// The number of thoughts tokens that the model should generate.
    pub thinking_budget: Option<u32>,
    /// Controls the maximum depth of the model's internal reasoning process
    /// before it produces a response. If not specified, the default is HIGH.
    /// Recommended for Gemini 3 or later models. Use with earlier models
    /// results in an error.
    pub thinking_level: Option<ThinkingLevel>,
}

/// Allow user to specify how much to think using enum instead of integer
/// budget.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ThinkingLevel {
    /// Unspecified thinking level.
    #[default]
    ThinkingLevelUnspecified,
    /// Minimal thinking level.
    Minimal,
    /// Low thinking level.
    Low,
    /// Medium thinking level.
    Medium,
    /// High thinking level.
    High,
    #[serde(other)]
    Other,
}

/// A response candidate generated from the model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {

    /// Generated content returned from the model.
    ///
    /// This field is not always populated, e.g.:
    ///
    /// ```json
    /// {"candidates": [{"finishReason": "UNEXPECTED_TOOL_CALL","index": 0}]}
    /// ```
    #[serde(default)]
    pub content: Option<Content>,
    /// The reason why the model stopped generating tokens. If empty, the model
    /// has not stopped generating tokens.
    pub finish_reason: Option<FinishReason>,
    /// List of ratings for the safety of a response candidate. There is at most
    /// one rating per category.
    pub safety_ratings: Option<Vec<SafetyRating>>,


    /// Citation information for model-generated candidate.
    ///
    /// This field may be populated with recitation information for any text
    /// included in the content. These are passages that are "recited" from
    /// copyrighted material in the foundational LLM's training data.
    #[serde(default)]
    pub citation_metadata: Option<CitationMetadata>,
    /// Token count for this candidate.
    pub token_count: Option<u32>,
    #[serde(default)]
    pub grounding_attributions: Vec<GroundingAttribution>,

    /// Grounding metadata for the candidate. This field is populated for
    /// `GenerateContent` calls.
    pub grounding_metadata: Option<GroundingMetadata>,
    /// Average log probability score of the candidate.
    pub avg_logprobs: Option<f32>,
    // TODO
    // /// Log-likelihood scores for the response tokens and top tokens
    // pub logprobs_result: Option<LogprobsResult>,
    // TODO
    // /// Metadata related to url context retrieval tool.
    // pub url_retrieval_metadata: Option<UrlRetrievalMetadata>,
    /// Metadata related to url context retrieval tool.
    pub url_context_metadata: Option<UrlContextMetadata>,
    /// Index of the candidate in the list of response candidates.
    pub index: Option<u32>,
}

/// Attribution for a source that contributed to an answer.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GroundingAttribution {
    /// Identifier for the source contributing to this attribution.
    #[serde(default)]
    pub source_id: Option<AttributionSourceId>,
    /// Grounding source content that makes up this attribution.
    pub content: Content,
}


/// Identifier for the source contributing to this attribution.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged, rename_all = "camelCase")]
pub enum AttributionSourceId {

    /// Identifier for an inline passage.
    GroundingPassage(GroundingPassageId),
    /// Identifier for a Chunk fetched via Semantic Retriever.
    SemanticRetrieverChunk(SemanticRetrieverChunk),
}


/// Identifier for a part within a `GroundingPassage`.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingPassageId {
    /// ID of the passage matching the `GenerateAnswerRequest`'s
    /// `GroundingPassage.id`.
    pub passage_id: Option<String>,
    /// Index of the part within the `GenerateAnswerRequest`'s
    /// `GroundingPassage.content`.
    pub part_index: Option<u32>,
}

/// Identifier for a Chunk fetched via Semantic Retriever specified in the
/// `GenerateAnswerRequest` using `SemanticRetrieverConfig`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct SemanticRetrieverChunk {
    /// Name of the source matching the request's
    /// `SemanticRetrieverConfig.source`.
    ///
    /// Example: `corpora/123` or `corpora/123/documents/abc`
    pub source: String,
    /// Name of the Chunk containing the attributed text.
    ///
    /// Example: `corpora/123/documents/abc/chunks/xyz`
    pub chunk: String,
}


#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    #[serde(default)]
    pub citation_sources: Vec<CitationSource>,
}

/// CitationSource
///
/// A citation to a source for a portion of a specific response.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CitationSource {
    /// Start of segment of the response that is attributed to this source.
    /// Index indicates the start of the segment, measured in bytes.
    pub start_index: Option<u32>,
    /// End of the attributed segment, exclusive.
    pub end_index: Option<u32>,
    /// URI that is attributed as a source for a portion of the text.
    pub uri: Option<String>,
    /// License for the GitHub project that is attributed as a source for
    /// segment. License info is required for code citations.
    pub license: Option<String>,
}

/// Safety rating for a piece of content.
///
/// The safety rating contains the category of harm and the harm probability
/// level in that category for a piece of content. Content is classified for
/// safety across a number of harm categories and the probability of the harm
/// classification is included here.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {

    /// The category for this rating.
    pub category: HarmCategory,
    /// The probability of harm for this content.
    pub probability: HarmProbability,
    /// Was this content blocked because of this rating?
    pub blocked: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmProbability {
    /// Default value. This value is unused.
    #[default]
    HarmProbabilityUnspecified,
    /// Content has a negligible chance of being unsafe.
    Negligible,
    /// Content has a low chance of being unsafe.
    Low,
    /// Content has a medium chance of being unsafe.
    Medium,
    /// Content has a high chance of being unsafe.
    High,
    #[serde(other)]
    Other,
}


// HarmCategory
//
// The category of a rating.
//
// These categories cover various kinds of harms that developers may wish to
// adjust.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    /// Default value. This value is unused.
    #[default]
    HarmCategoryUnspecified,
    Derogatory,
    Toxicity,
    Violence,
    Sexual,
    Medical,
    Dangerous,
    Harassment,
    HateSpeech,
    SexuallyExplicit,
    DangerousContent,
    CivicIntegrity,
    #[serde(other)]
    Other,
}


/// Metadata returned to client when grounding is enabled.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    /// List of supporting references retrieved from specified grounding source.
    #[serde(default)]
    pub grounding_chunks: Vec<GroundingChunk>,
    /// List of grounding support.
    #[serde(default)]
    pub grounding_supports: Vec<GroundingSupport>,
    /// Web search queries for the following-up web search.
    #[serde(default)]
    pub web_search_queries: Vec<String>,
    /// Optional. Google search entry for the following-up web searches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_entry_point: Option<SearchEntryPoint>,
    /// Metadata related to retrieval in the grounding flow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_metadata: Option<RetrievalMetadata>,
}

/// Grounding chunk.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum GroundingChunk {
    /// Grounding chunk from the web.
    Web(Web),
}

/// Chunk from the web
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Web {
    /// URI reference of the chunk
    pub uri: String,
    /// Title of the chunk
    pub title: String,
}

/// Grounding support.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    /// A list of indices (into 'grounding_chunk') specifying the citations associated with the claim.
    /// For instance [1,3,4] means that grounding_chunk[1], grounding_chunk[3], grounding_chunk[4] are the
    /// retrieved content attributed to the claim.
    #[serde(default)]
    pub grounding_chunk_indices: BTreeSet<u32>,
    /// Confidence score of the support references. Ranges from 0 to 1. 1 is the most confident.
    /// This list must have the same size as the groundingChunkIndices.
    #[serde(default)]
    pub confidence_scores: Vec<f64>,
    /// Segment of the content this support belongs to.
    pub segment: Segment,
}

/// Segment of the content.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    /// Output only. The index of a Part object within its parent Content object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part_index: Option<u32>,
    /// Output only. Start index in the given Part, measured in bytes. Offset from the start of the Part, inclusive, starting at zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u32>,
    /// Output only. End index in the given Part, measured in bytes. Offset from the start of the Part, exclusive, starting at zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<u32>,
    /// Output only. The text corresponding to the segment from the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Google search entry point.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    /// Optional. Web content snippet that can be embedded in a web page or an app webview.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_content: Option<String>,
    /// Optional. Base64 encoded JSON representing array of <search term, search url> tuple.
    /// A base64-encoded string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_blob: Option<String>,
}

/// Retrieval metadata for grounding
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalMetadata {
    /// Optional. Score indicating how likely information from google search could help answer the prompt.
    /// The score is in the range [0, 1], where 0 is the least likely and 1 is the most likely.
    /// This score is only populated when google search grounding and dynamic retrieval is enabled.
    /// It will be compared to the threshold to determine whether to trigger google search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_dynamic_retrieval_score: Option<f64>,
}

/// Metadata related to url context retrieval tool.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UrlContextMetadata {
    /// List of url context.
    #[serde(default)]
    pub url_metadata: Vec<UrlMetadata>,
}

/// Context of the a single url retrieval.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct UrlMetadata {
    /// Retrieved url by the tool.
    pub retrieved_url: String,
    /// Status of the url retrieval.
    pub url_retrieval_status: UrlRetrievalStatus,
}

/// Status of the url retrieval.
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum UrlRetrievalStatus {
    /// Default value. This value is unused.
    #[default]
    UrlRetrievalStatusUnspecified,
    /// Url retrieval is successful.
    Success,
    /// Url retrieval is failed due to error.
    Error,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    /// Default value. This value is unused.
    #[default]
    FinishReasonUnspecified,
    /// Natural stop point of the model or provided stop sequence.
    Stop,
    /// The maximum number of tokens as specified in the request was reached.
    MaxTokens,
    /// The response candidate content was flagged for safety reasons.
    Safety,
    /// The response candidate content was flagged for recitation reasons.
    Recitation,
    /// The response candidate content was flagged for using an unsupported
    /// language.
    Language,
    /// Unknown reason.
    Other,
    /// Token generation stopped because the content contains forbidden terms.
    Blocklist,
    /// Token generation stopped for potentially containing prohibited content.
    ProhibitedContent,
    /// Token generation stopped because the content potentially contains
    /// Sensitive Personally Identifiable Information (SPII).
    Spii,
    /// The function call generated by the model is invalid.
    MalformedFunctionCall,
    /// Token generation stopped because generated images contain safety
    /// violations.
    ImageSafety,
    /// Model generated a tool call but no tools were enabled in the request.
    UnexpectedToolCall,
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged, rename_all = "camelCase")]
pub enum Part {
    /// Standard text part.
    Text { 
        text: String 
    },
    /// Thought / Reasoning part (Gemini 3).
    Thought {
        text: String,
        #[serde(default)]
        thought: bool,
    },
    /// Inline binary data.
    InlineData {
        #[serde(rename = "inlineData")]
        data: InlineData
    },
    /// Data stored in a file (e.g. via File API).
    FileData {
        #[serde(rename = "fileData")]
        data: FileData
    },
    /// A call to a tool/function.
    FunctionCall {
        #[serde(rename = "functionCall")]
        call: FunctionCall
    },
    /// A response from a tool/function.
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        response: FunctionResponse
    },
    /// Executable code (e.g. Python for code execution).
    ExecutableCode {
        #[serde(rename = "executableCode")]
        code: ExecutableCode
    },
    /// Result of code execution.
    CodeExecutionResult {
        #[serde(rename = "codeExecutionResult")]
        result: Value
    },
    /// Opaque thought signature for stateful reasoning (Gemini 3).
    ThoughtSignature {
        #[serde(rename = "thoughtSignature")]
        signature: String
    },
}

impl Part {
    pub fn text(t: impl Into<String>) -> Self {
        Self::Text { text: t.into() }
    }

    pub fn thought(t: impl Into<String>) -> Self {
        Self::Thought {
            text: t.into(),
            thought: true,
        }
    }
    
    pub fn inline_data(mime_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::InlineData {
            data: InlineData {
                mime_type: mime_type.into(),
                data: data.into(),
            }
        }
    }

    pub fn file_data(mime_type: impl Into<String>, file_uri: impl Into<String>) -> Self {
        Self::FileData {
            data: FileData {
                mime_type: mime_type.into(),
                file_uri: file_uri.into(),
            }
        }
    }
}


/// [DEPRECATED] Use [Part] instead.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
#[allow(deprecated)]
#[deprecated(since = "0.10.0", note = "Use Part instead")]
pub struct ContentPart {


    #[serde(default, skip_serializing_if = "is_false")]
    pub thought: bool,
    #[allow(deprecated)]
    #[serde(flatten)]
    pub data: ContentData,

    #[serde(skip_serializing)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[allow(deprecated)]
impl From<Part> for ContentPart {
    fn from(part: Part) -> Self {
        match part {
            Part::Text { text } => ContentPart::new_text(&text, false),
            Part::InlineData { data } => ContentPart::new_inline_data(&data.mime_type, &data.data, false),
            Part::FileData { data } => ContentPart::new_file_data(&data.mime_type, &data.file_uri),
            Part::FunctionCall { call } => ContentPart::new_function_call(call.id.as_deref(), &call.name, call.arguments, false),
            Part::FunctionResponse { response } => ContentPart::new_function_response(response.id.as_deref(), &response.name, response.response.content),
            Part::ExecutableCode { code } => ContentPart::new_executable_code(&code.code),
            Part::CodeExecutionResult { result } => ContentPart::new_code_execution_result(result),
            Part::Thought { text, .. } => ContentPart::new_text(&text, true),
            Part::ThoughtSignature { signature } => {
                let mut cp = ContentPart::new_text("", false);
                cp.thought_signature = Some(signature);
                cp
            }
        }
    }
}

#[allow(deprecated)]
impl From<ContentPart> for Part {


    fn from(cp: ContentPart) -> Self {
        if cp.thought {
            if let ContentData::Text(t) = cp.data {
                return Part::Thought { text: t, thought: true };
            }
        }
        if let Some(sig) = cp.thought_signature {
            return Part::ThoughtSignature { signature: sig };
        }
        match cp.data {
            ContentData::Text(t) => Part::Text { text: t },
            ContentData::InlineData(d) => Part::InlineData { data: d },
            ContentData::FileData(d) => Part::FileData { data: d },
            ContentData::FunctionCall(c) => Part::FunctionCall { call: c },
            ContentData::FunctionResponse(r) => Part::FunctionResponse { response: r },
            ContentData::ExecutableCode(c) => Part::ExecutableCode { code: c },
            ContentData::CodeExecutionResult(v) => Part::CodeExecutionResult { result: v },
        }
    }
}


#[allow(deprecated)]
impl ContentPart {
    pub fn new_text(text: &str, thought: bool) -> Self {
        Self {
            data: ContentData::Text(text.to_string()),
            thought,
            metadata: None,
            thought_signature: None,
        }
    }

    pub fn new_inline_data(mime_type: &str, data: &str, thought: bool) -> Self {
        Self {
            data: ContentData::InlineData(InlineData {
                mime_type: mime_type.to_string(),
                data: data.to_string(),
            }),
            thought,
            metadata: None,
            thought_signature: None,
        }
    }

    pub fn new_file_data(mime_type: &str, file_uri: &str) -> Self {
        Self {
            data: ContentData::FileData(FileData {
                mime_type: mime_type.to_string(),
                file_uri: file_uri.to_string(),
            }),
            thought: false,
            metadata: None,
            thought_signature: None,
        }
    }

    pub fn new_function_call(
        id: Option<&str>,
        name: &str,
        arguments: Value,
        thought: bool,
    ) -> Self {
        Self {
            data: ContentData::FunctionCall(FunctionCall {
                id: id.map(|s| s.to_string()),
                name: name.to_string(),
                arguments,
            }),
            thought,
            metadata: None,
            thought_signature: None,
        }
    }

    pub fn new_executable_code(code: &str) -> Self {
        Self {
            data: ContentData::ExecutableCode(ExecutableCode {
                code: code.to_string(),
            }),
            thought: false,
            metadata: None,
            thought_signature: None,
        }
    }

    pub fn new_code_execution_result(content: Value) -> Self {
        Self {
            data: ContentData::CodeExecutionResult(content),
            thought: false,
            metadata: None,
            thought_signature: None,
        }
    }

    pub fn new_function_response(id: Option<&str>, name: &str, content: Value) -> Self {
        Self {
            data: ContentData::FunctionResponse(FunctionResponse {
                id: id.map(|s| s.to_string()),
                name: name.to_string(),
                response: FunctionResponsePayload { content },
            }),
            thought: false,
            metadata: None,
            thought_signature: None,
        }
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// [DEPRECATED] Use [Part] instead.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
#[allow(deprecated)]
#[deprecated(since = "0.10.0", note = "Use Part instead")]
pub enum ContentData {


    Text(String),
    InlineData(InlineData),
    FileData(FileData),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
    ExecutableCode(ExecutableCode),
    CodeExecutionResult(Value),
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCall {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default, rename = "args")]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    pub response: FunctionResponsePayload,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResponsePayload {
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableCode {
    pub code: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    // FIXME: This should be part of the API response, but it's not.
    //
    // See:
    // <https://discuss.ai.google.dev/t/basemodelid-is-not-available-in-api-response/55268>
    #[serde(skip_deserializing)]
    pub base_model_id: String,
    pub version: String,
    pub display_name: String,
    pub description: Option<String>,
    pub input_token_limit: u32,
    pub output_token_limit: u32,
    pub supported_generation_methods: Vec<String>,
    pub temperature: Option<f32>,
    pub max_temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct File {
    pub name: String,
    pub display_name: Option<String>,
    pub mime_type: String,
    pub size_bytes: String,
    pub create_time: String,
    pub update_time: String,
    pub expiration_time: String,
    pub sha256_hash: String,
    pub uri: String,
    pub state: FileState,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FileState {
    #[default]
    StateUnspecified,
    Processing,
    Active,
    Failed,
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::{
        FunctionDeclaration, SchemaType,
    };



    #[test]
    fn function_declaration_serialization() {
        use super::Schema;
        let declaration = FunctionDeclaration {
            name: "lookup_status".to_string(),
            description: "Looks up service status".to_string(),
            parameters: Some(Schema {
                schema_type: SchemaType::Object,
                properties: Some(HashMap::from([(
                    "service".to_string(),
                    Schema {
                        schema_type: SchemaType::String,
                        description: Some("Service name".to_string()),
                        ..Default::default()
                    },
                )])),
                required: Some(vec!["service".to_string()]),
                ..Default::default()
            }),


            response: None,
        };


        let serialized = serde_json::to_value(&declaration).expect("declaration should serialize");
        let object = serialized
            .as_object()
            .expect("function declaration should serialize to an object");

        assert!(object.contains_key("parameters"));
        assert!(!object.contains_key("parametersJsonSchema"));
        assert!(!object.contains_key("response"));
        assert_eq!(
            object
                .get("parameters")
                .and_then(|value| value.get("required")),
            Some(&json!(["service"]))
        );
    }
}

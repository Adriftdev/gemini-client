#[cfg(feature = "tracing")]
use crate::GeminiError;

#[cfg(feature = "tracing")]
#[allow(dead_code)]
pub(crate) struct SpanGuard(Option<tracing::Span>);

#[cfg(not(feature = "tracing"))]
pub(crate) struct SpanGuard;

impl SpanGuard {
    #[cfg(feature = "tracing")]
    pub(crate) fn from_info_span(span: tracing::Span) -> Self {
        Self(Some(span))
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn from_debug_span(span: tracing::Span) -> Self {
        Self(Some(span))
    }

    #[cfg(not(feature = "tracing"))]
    pub(crate) fn from_info_span() -> Self {
        Self
    }

    #[cfg(not(feature = "tracing"))]
    pub(crate) fn from_debug_span() -> Self {
        Self
    }
}

#[cfg(feature = "tracing")]
pub(crate) fn gemini_error_kind(error: &GeminiError) -> &'static str {
    match error {
        GeminiError::Http(_) => "http",
        GeminiError::EventSource(_) => "event_source",
        GeminiError::Api(_) => "api",
        GeminiError::Json { .. } => "json",
        GeminiError::FunctionExecution(_) => "function_execution",
        GeminiError::LoopLimitExceeded { .. } => "loop_limit_exceeded",
    }
}

macro_rules! telemetry_span_guard {
    (info, $name:expr $(, $($field:tt)+)?) => {{
        #[cfg(feature = "tracing")]
        {
            $crate::telemetry::SpanGuard::from_info_span(
                tracing::info_span!($name $(, $($field)+)?)
            )
        }
        #[cfg(not(feature = "tracing"))]
        {
            $crate::telemetry::SpanGuard::from_info_span()
        }
    }};
    (debug, $name:expr $(, $($field:tt)+)?) => {{
        #[cfg(feature = "tracing")]
        {
            $crate::telemetry::SpanGuard::from_debug_span(
                tracing::debug_span!($name $(, $($field)+)?)
            )
        }
        #[cfg(not(feature = "tracing"))]
        {
            $crate::telemetry::SpanGuard::from_debug_span()
        }
    }};
}

macro_rules! telemetry_info {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::info!($($tt)*);
        }
    }};
}

macro_rules! telemetry_debug {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::debug!($($tt)*);
        }
    }};
}

macro_rules! telemetry_warn {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::warn!($($tt)*);
        }
    }};
}

macro_rules! telemetry_error {
    ($($tt:tt)*) => {{
        #[cfg(feature = "tracing")]
        {
            tracing::error!($($tt)*);
        }
    }};
}

pub(crate) use telemetry_debug;
pub(crate) use telemetry_error;
pub(crate) use telemetry_info;
pub(crate) use telemetry_span_guard;
pub(crate) use telemetry_warn;

# Contributing to gemini-client-rs

Thank you for your interest in contributing to `gemini-client-rs`. This project provides a high-performance, idiomatic, and infrastructure-first Rust client for the Google Gemini API.

We focus on being a **precision thin layer**—prioritizing raw API fidelity, reliability, and architectural clarity.

---

## 🏗️ Getting Started

### Prerequisites

- **Rust**: Latest stable version (Edition 2021).
- **Google AI Studio API Key**: Required for running integration tests and examples.

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Adriftdev/gemini-client.git
   cd gemini-client
   ```

2. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Run Examples**:
   Verify your setup by running the basic example:
   ```bash
   cargo run --example basic
   ```

---

## 🏛️ Ecosystem Alignment

This library is part of the `rain` ecosystem. To maintain consistency across clients (e.g., `ollama-client-rs`), we adhere to a shared set of interface patterns:

1. **Macro-First DX**: Prioritize declarative request building via procedural macros (`gemini_chat!`, `gemini_parts!`).
2. **Standardized Schema Generation**: All structured outputs should use the `GeminiSchema` derive macro.
3. **Builder Pattern**: Clients should always implement `new`, `with_client`, and `with_api_url`.
4. **Pinned Streams**: All streaming methods must return `Pin<Box<dyn Stream>>` to simplify caller integration.
5. **Standardized Telemetry**: Use the internal `telemetry_*!` macros.
6. **Error Mapping**: Maintain a flat, descriptive `GeminiError` enum using `thiserror`.

---

## 🛠️ Architectural Philosophy

When contributing to the core client, adhere to these principles:

1. **Thin Layer Foundation**: The core client is a transport and mapping layer. Avoid adding complex state machines or orchestration logic.
2. **Ergonomic Macro Overlay**: Provide a best-in-class developer experience via macros that simplify common tasks (e.g., chat construction, tool declaration).
3. **Transparent Proxy**: Preserving byte-for-byte fidelity and API structure in the core types is a priority.
4. **Rust Type Safety**: Leverage Rust's type system to make API constraints explicit and compile-time safe.
5. **Zero-Overhead Abstractions**: Ensure the mapping from macros to request structs is efficient.


---

## 💻 Coding Standards

### 1. Idiomatic Rust
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use `thiserror` for all library-level error definitions.
- Avoid `anyhow` in the core library; it is reserved for examples and tests.

### 2. Linting & Formatting
We maintain strict quality gates. PRs will not be merged if they contain Clippy warnings.
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

### 3. Telemetry
Always use the standardized macros to ensure consistent mapping of error kinds and span data for cloud-native observability.

---

## 🧪 Testing Protocol

Every PR must be verified across multiple feature configurations:

1. **Core Tests**:
   ```bash
   cargo test
   ```

2. **Feature-Specific Tests**:
   Verify the `tracing` feature and other optional dependencies:
   ```bash
   cargo test --features tracing
   ```

3. **Example Verification**:
   Ensure basic examples remain functional:
   ```bash
   cargo check --examples
   ```

---

## 📬 Contribution Workflow

1. **Open an Issue**: Discuss major changes before implementation.
2. **Fork and Branch**: Work on a feature branch (`feat/your-feature` or `fix/your-fix`).
3. **Conventional Commits**: We use conventional commit messages (e.g., `feat: add embedding support`).
4. **PR Review**: All PRs require approval and passing CI checks.

---

## 🛡️ Security

- **Never commit your `.env` file or API keys**.
- Report security vulnerabilities via GitHub Issues or contact the maintainers directly.

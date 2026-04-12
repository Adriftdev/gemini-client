# Contributing to gemini-client-rs

Thank you for your interest in contributing to `gemini-client-rs`. This project aims to provide a high-performance, idiomatic Rust client for the Google Gemini API, with a specialized `agentic` layer for deterministic multi-agent orchestration.

As a systems-oriented project, we prioritize **reliability, determinism, and architectural clarity**.

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
   Verify your setup by running a basic example:
   ```bash
   cargo run --example basic
   ```

---

## 🛠️ Architectural Philosophy

When contributing to the `agentic` module or core client, adhere to these principles:

1. **Occam's Razor**: Prefer the simplest implementation that satisfies the requirements. Avoid over-engineering orchestrators unless scale or resilience demands it.
2. **Deterministic Orchestration**: Higher-level patterns (Supervisor, Worker, etc.) should have predictable state transitions. Avoid hidden side effects in agent loops.
3. **Store and Forward (Persistence)**: For agentic workflows requiring long-running state, utilize local disk-backed buffers or persistent blackboards.
4. **Transparent Proxy**: Ensure the low-level client remains a clean relay for the Gemini API, preserving byte-for-byte fidelity where possible.

---

## 💻 Coding Standards

### 1. Idiomatic Rust
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use `anyhow` for applications and examples.
- Use `thiserror` for library-level error definitions in `src/`.

### 2. Linting & Formatting
We maintain strict quality gates. PRs will not be merged if they contain Clippy warnings.
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

### 3. Concurrency
- Use `tokio` for async operations.
- Prefer ownership and message passing over shared mutable state.
- When shared state is necessary, use `Arc<RwLock<T>>` or `Arc<Mutex<T>>`.

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
   If you modify the `agentic` module, you MUST verify the corresponding examples:
   ```bash
   cargo check --examples
   cargo run --example supervisor_workflow
   ```

---

## 📬 Contribution Workflow

1. **Open an Issue**: Discuss major changes before implementation.
2. **Fork and Branch**: Work on a feature branch (`feat/your-feature` or `fix/your-fix`).
3. **Conventional Commits**: We recommend using conventional commit messages (e.g., `feat: add RAG caching`).
4. **PR Review**: All PRs require at least one approval. Ensure all CI checks pass.

---

## 🛡️ Security

- **Never commit your `.env` file or API keys**.
- If you find a security vulnerability, please report it via GitHub Issues (marking as private if possible) or contact the maintainers directly.

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// PersonaPlex adaptations by Monetic Labs.

//! moshi-backend library interface
//!
//! Exposes the streaming server components for use as a library dependency.
//! The voice-server in an-ecosystem imports these to build a managed
//! PersonaPlex backend without reimplementing the streaming pipeline.
//!
//! # Key types
//!
//! - [`stream_both::AppStateInner`] — loaded model state (LM, Mimi, tokenizer)
//! - [`stream_both::StreamingModel`] — per-session streaming wrapper
//! - [`stream_both::handle_socket`] — WebSocket handler (Opus↔Mimi↔LM↔Mimi↔Opus)
//! - [`standalone::device`] — Metal/CUDA/CPU device selection

pub mod audio;
pub mod benchmark;
pub mod standalone;
pub mod stream_both;
pub mod utils;

/// Arguments for standalone mode (used by standalone.rs and benchmark.rs)
#[derive(Clone, Debug)]
pub struct StandaloneArgs {
    pub cpu: bool,
}

/// Arguments for benchmark mode
#[derive(Clone, Debug)]
pub struct BenchmarkArgs {
    pub cpu: bool,
    pub steps: usize,
    pub reps: usize,
    pub stat_file: Option<String>,
    pub chrome_tracing: bool,
    pub asr: bool,
    pub mimi_only: bool,
}

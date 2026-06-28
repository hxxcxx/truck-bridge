//! # truck-bridge
//!
//! C ABI export of the [truck] geometric/CAD kernel.
//!
//! This crate exposes truck's types as a small, stable C ABI. It produces a
//! dynamic library (`truck_bridge.dll` / `.so` / `.dylib`) and a static library,
//! plus a generated C header at `include/truck_bridge.h`.
//!
//! ## Design invariants (the "four foundations")
//!
//! These are established once here and reused by every exported type:
//!
//! 1. **Error model** — see [`error`]. Functions that can fail return a `bool`
//!    (or a nullable handle) and optionally report details via a `TruckError`
//!    handle. Panics never cross the FFI boundary: every `extern "C"` body runs
//!    under a `catch_unwind` guard.
//! 2. **Ownership** — see [`handle`]. Truck objects are held behind *opaque
//!    handles* (C never sees their layout). Returned arrays/strings are
//!    `{ptr, len}` views; each comes with a matching `*_free`.
//! 3. **ABI version** — see [`version`]. `truck_abi_version()` lets consumers
//!    reject a mismatched `.dll` at load time.
//! 4. **Generated header** — `build.rs` regenerates `truck_bridge.h` via
//!    cbindgen; never hand-edit it.
//!
//! [truck]: https://github.com/ricosjp/truck
//!
//! ## Out of scope
//!
//! C++ (or other-language) RAII wrappers are deliberately NOT provided here.
//! Consumers build their own thin wrapper on top of the C header so that their
//! preferred idioms (smart pointers, exceptions/optionals, stdlib choices) and
//! compiler are not constrained by this library.

#![deny(unsafe_op_in_unsafe_fn)]
// Keep clippy quiet without being militant; we relax a couple that fight FFI.
#![allow(clippy::missing_safety_doc)]
// `missing_docs` is enforced at the module level for public FFI items.

pub mod error;
pub mod handle;
pub mod polymesh;
pub mod topology;
pub mod version;

// Re-export the concrete truck type used by polymesh so the whole crate agrees
// on the single monomorphized form of `PolygonMesh<V, A>`.
pub(crate) use truck_polymesh::PolygonMesh;

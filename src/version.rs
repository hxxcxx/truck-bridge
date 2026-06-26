//! ABI versioning.
//!
//! ## Foundation #3 — reject a mismatched `.dll` at load time.
//!
//! The library's version is a build-time constant. Consumers should assert at
//! startup that `truck_abi_version()` matches the `TRUCK_BRIDGE_ABI_VERSION`
//! macro their compiled-in header carries, so that linking against a stale
//! library (whose opaque-handle layout may differ) fails loudly instead of
//! corrupting memory.

use crate::handle::TruckStr;

/// Bumped on any change that could alter the C-visible layout or semantics of
/// exported items. Stays in lockstep with the `TRUCK_BRIDGE_ABI_VERSION` macro
/// emitted by cbindgen into the header.
pub const TRUCK_BRIDGE_ABI_VERSION: u32 = 1;

/// Human-readable version string. Composed at compile time from Cargo env vars.
const VERSION_STRING: &str = concat!(
    "truck-bridge ",
    env!("CARGO_PKG_VERSION"),
    " / truck-polymesh 0.6.0",
);

/// Returns the ABI version. Consumers compare this to the
/// `TRUCK_BRIDGE_ABI_VERSION` macro in their copy of the header.
///
/// The returned `u32` is a plain value type — no ownership concerns.
#[no_mangle]
pub extern "C" fn truck_abi_version() -> u32 {
    TRUCK_BRIDGE_ABI_VERSION
}

/// Returns a build/version descriptor string (UTF-8 bytes, NOT NUL-terminated).
/// Free it with [`crate::handle::truck_str_free`].
#[no_mangle]
pub extern "C" fn truck_version_string() -> TruckStr {
    TruckStr::from(VERSION_STRING.as_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_one() {
        assert_eq!(truck_abi_version(), 1);
        assert_eq!(TRUCK_BRIDGE_ABI_VERSION, 1);
    }

    #[test]
    fn version_string_is_freed_safely() {
        let s = truck_version_string();
        assert!(s.len > 0);
        // SAFETY: s came from truck_version_string.
        unsafe { crate::handle::truck_str_free(s) };
    }
}

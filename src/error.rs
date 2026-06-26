//! Error model for the C ABI.
//!
//! ## Foundation #1 — how failures cross the boundary.
//!
//! Functions that can fail:
//!   - return a `bool` (`false` = failure) or a nullable handle,
//!   - and, for detailed failures, write a [`TruckError`] handle into an
//!     `*mut *mut TruckError` out-parameter.
//!
//! Panics from truck or our own glue **never** unwind across the FFI boundary.
//! Every `extern "C"` body wraps its real work in [`catch`] (or the
//! [`truck_guard`] macro) which converts a panic into a [`TruckError`].
//!
//! The types and functions in this module are intentionally written with plain
//! signatures (no `FnOnce(...) -> ...` bounds, no `dyn A + B`) so that
//! cbindgen's parser can scan the file without choking.

use crate::handle::{self, TruckStr};

/// Opaque error handle. C sees only `typedef struct TruckError TruckError;`.
///
/// Holds a single UTF-8 message string. Allocated by this library, freed via
/// [`truck_error_free`].
#[derive(Debug)]
pub struct TruckError {
    message: String,
}

impl TruckError {
    /// Create a new error from any message.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        TruckError { message: message.into() }
    }

    /// The error message, as a `&str`.
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

/// Result of a guarded FFI body: success value, or an owning error handle.
/// Kept as a named type so the contract reads uniformly across modules even
/// though the `truck_guard!` macro currently inlines it.
#[allow(dead_code)]
pub(crate) type GuardResult<T> = Result<T, *mut TruckError>;

/// Wrap an `extern "C"` body so a panic becomes a [`TruckError`] instead of
/// unwinding across the FFI boundary.
///
/// Usage:
/// ```ignore
/// #[no_mangle]
/// pub unsafe extern "C" fn truck_foo(h: *mut Thing, err: *mut *mut TruckError) -> bool {
///     let res = truck_guard!(|| {
///         let t = handle::from_ref(h).ok_or_else(|| TruckError::new("null handle"))?;
///         do_work(t)
///     });
///     finish(res, err)
/// }
/// ```
///
/// Implemented as a `macro_rules!` macro (which cbindgen skips) rather than a
/// generic function, so this file has no `FnOnce() -> ...` bound that would
/// break cbindgen's parser.
macro_rules! truck_guard {
    ($body:expr) => {{
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe($body)) {
            ::std::result::Result::Ok(::std::result::Result::Ok(v)) =>
                ::std::result::Result::Ok(v),
            ::std::result::Result::Ok(::std::result::Result::Err(e)) =>
                ::std::result::Result::Err($crate::handle::into_raw(e)),
            ::std::result::Result::Err(payload) => {
                let msg = $crate::error::panic_to_string(&payload);
                ::std::result::Result::Err($crate::handle::into_raw(
                    $crate::error::TruckError::new(msg),
                ))
            }
        }
    }};
}
pub(crate) use truck_guard;

/// Turn a panic payload (`Box<dyn Any + Send>`) into a best-effort string.
///
/// Written as a plain function taking a trait object so cbindgen can parse it.
pub(crate) fn panic_to_string(payload: &Box<dyn std::any::Any + Send>) -> String {
    use std::any::Any;
    // `payload` is `&Box<dyn Any + Send>`; reach the inner `dyn Any` so the
    // downcasts below match the value the panic was constructed with, not the
    // Box wrapper itself.
    let any: &dyn Any = &**payload;
    if let Some(s) = any.downcast_ref::<&'static str>() {
        format!("internal panic: {}", s)
    } else if let Some(s) = any.downcast_ref::<String>() {
        format!("internal panic: {}", s)
    } else {
        "internal panic: <non-string payload>".to_string()
    }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// Copy the error message as a UTF-8 byte string.
///
/// The returned [`TruckStr`] is a fresh allocation owned by the caller; free it
/// with [`truck_str_free`]. Passing `NULL` yields an empty `TruckStr`.
///
/// # Safety
/// `err` must be either NULL or a valid pointer returned by a truck-bridge
/// function that produced an error handle.
#[no_mangle]
pub unsafe extern "C" fn truck_error_message(err: *const TruckError) -> TruckStr {
    // SAFETY: caller guarantees err is NULL or a valid TruckError handle.
    match unsafe { handle::from_ref(err) } {
        Some(e) => TruckStr::from(e.message().as_bytes().to_vec()),
        None => TruckStr::empty(),
    }
}

/// Free an error handle. Idempotent: `truck_error_free(NULL)` is a no-op.
///
/// # Safety
/// `err` must be NULL or a pointer previously returned by a truck-bridge error
/// out-parameter, and must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_error_free(err: *mut TruckError) {
    // SAFETY: caller guarantees err is NULL or a valid, owned handle.
    match unsafe { handle::take_raw(err) } {
        Some(e) => drop(e),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_guarded<F: FnOnce() -> Result<i32, TruckError>>(f: F) -> GuardResult<i32> {
        truck_guard!(f)
    }

    #[test]
    fn guard_catches_panic() {
        let res = run_guarded(|| -> Result<i32, TruckError> {
            panic!("boom");
        });
        let err_ptr = res.expect_err("panic must become Err");
        // SAFETY: err_ptr came from guard(); we own it.
        let msg = unsafe { &*err_ptr }.message();
        assert!(msg.contains("internal panic"), "got: {msg}");
        assert!(msg.contains("boom"), "got: {msg}");
        // SAFETY: freeing our own handle.
        unsafe { truck_error_free(err_ptr) };
    }

    #[test]
    fn guard_propagates_logical_error() {
        let res = run_guarded(|| Err(TruckError::new("parse failed")));
        let err_ptr = res.expect_err("logical error must be Err");
        // SAFETY: we own err_ptr.
        assert_eq!(unsafe { &*err_ptr }.message(), "parse failed");
        unsafe { truck_error_free(err_ptr) };
    }

    #[test]
    fn guard_ok_on_success() {
        let res = run_guarded(|| Ok::<_, TruckError>(42));
        assert_eq!(res.unwrap(), 42);
    }

    #[test]
    fn error_message_null_yields_empty() {
        // SAFETY: NULL is explicitly allowed.
        let s = unsafe { truck_error_message(std::ptr::null()) };
        assert_eq!(s.len, 0);
        // SAFETY: s is an empty (null ptr) TruckStr; free tolerates it.
        unsafe { crate::handle::truck_str_free(s) };
    }
}

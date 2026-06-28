//! Ownership primitives for the C ABI.
//!
//! ## Foundation #2 — who owns what across the boundary.
//!
//! - **Opaque handles** for truck objects: C gets `typedef struct X X;` and only
//!   ever touches `X*`. The real type lives in the Rust allocator and is reached
//!   via [`into_raw`] / [`take_raw`] / [`from_ref`] / [`from_mut`].
//! - **Owned byte/number arrays** (`TruckF64Array`, `TruckF32Array`,
//!   `TruckU8Array`, `TruckU32Array`, `TruckStr`): a `{ptr, len}` view into
//!   Rust-allocated memory handed to C. The caller must release it with the
//!   matching `*_free`. Never `free()`/`delete` these — they belong to the Rust
//!   allocator.

use std::mem::ManuallyDrop;

// ---------------------------------------------------------------------------
// Opaque handle helpers
// ---------------------------------------------------------------------------

/// Box a value and hand C a raw owning pointer. The pointer must later come
/// back through [`take_raw`] (or the matching `*_free`).
pub(crate) fn into_raw<T>(t: T) -> *mut T {
    Box::into_raw(Box::new(t))
}

/// Reclaim ownership of a pointer produced by [`into_raw`]. Returns `None` for
/// NULL.
///
/// # Safety
/// `p` must be NULL, or a pointer previously produced by [`into_raw`] that has
/// not yet been reclaimed.
pub(crate) unsafe fn take_raw<T>(p: *mut T) -> Option<T> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller guarantees p is a valid, owned, non-null Box origin.
    Some(*unsafe { Box::from_raw(p) })
}

/// Borrow a const pointer. Returns `None` for NULL.
///
/// # Safety
/// `p` must be NULL or a valid pointer to a live `T` that outlives the returned
/// reference.
pub(crate) unsafe fn from_ref<'a, T>(p: *const T) -> Option<&'a T> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller guarantees validity and lifetime.
    Some(unsafe { &*p })
}

/// Borrow a mut pointer. Returns `None` for NULL.
///
/// # Safety
/// `p` must be NULL or a valid, uniquely-reachable pointer to a live `T` that
/// outlives the returned reference.
pub(crate) unsafe fn from_mut<'a, T>(p: *mut T) -> Option<&'a mut T> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller guarantees validity, uniqueness, and lifetime.
    Some(unsafe { &mut *p })
}

// ---------------------------------------------------------------------------
// Owned array / string views handed to C
// ---------------------------------------------------------------------------

/// Owned `f64` array handed to C. Free with [`truck_f64array_free`].
#[repr(C)]
#[derive(Debug)]
pub struct TruckF64Array {
    /// Pointer to the first element (NULL iff `len == 0`).
    pub ptr: *mut f64,
    /// Number of elements.
    pub len: usize,
}

/// Owned `f32` array handed to C. Free with [`truck_f32array_free`].
#[repr(C)]
#[derive(Debug)]
pub struct TruckF32Array {
    pub ptr: *mut f32,
    pub len: usize,
}

/// Owned `u8` byte array handed to C. Free with [`truck_u8array_free`].
#[repr(C)]
#[derive(Debug)]
pub struct TruckU8Array {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Owned `u32` array handed to C. Free with [`truck_u32array_free`].
#[repr(C)]
#[derive(Debug)]
pub struct TruckU32Array {
    pub ptr: *mut u32,
    pub len: usize,
}

/// Owned UTF-8 byte string handed to C. **Not NUL-terminated.**
/// Free with [`truck_str_free`].
#[repr(C)]
#[derive(Debug)]
pub struct TruckStr {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Split a `Vec<T>` into `(ptr, len, capacity)` without running its destructor,
/// transferring ownership to the caller.
pub(crate) fn vec_into_raw_parts<T>(mut v: Vec<T>) -> (*mut T, usize, usize) {
    // We need both the allocation pointer and the original capacity so the
    // matching `*_free` can reconstruct and drop the Vec exactly.
    let ptr = v.as_mut_ptr();
    let len = v.len();
    let cap = v.capacity();
    let _ = ManuallyDrop::new(v);
    (ptr, len, cap)
}

/// Reclaim a `{ptr,len}` view back into a `Vec<T>` so it can be dropped.
///
/// # Safety
/// `ptr/len/cap` must describe a valid allocation previously produced by
/// [`vec_into_raw_parts`], or `ptr` may be NULL with `len == cap == 0`.
pub(crate) unsafe fn vec_from_raw_parts<T>(ptr: *mut T, len: usize, cap: usize) -> Vec<T> {
    if ptr.is_null() {
        return Vec::new();
    }
    // SAFETY: caller guarantees the parts come from a matching Vec.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Helper: produce a `(ptr, len)` view from a Vec after shrinking it so that
/// capacity == len (the free path always recovers cap == len).
fn shrinked_view<T>(mut v: Vec<T>) -> (*mut T, usize) {
    v.shrink_to_fit();
    debug_assert_eq!(v.len(), v.capacity());
    let (ptr, len, _cap) = vec_into_raw_parts(v);
    (ptr, len)
}

impl TruckF64Array {
    /// Wrap a `Vec<f64>`, transferring its allocation to the caller (C).
    pub(crate) fn from(v: Vec<f64>) -> Self {
        let (ptr, len) = shrinked_view(v);
        Self { ptr, len }
    }
}

impl TruckF32Array {
    pub(crate) fn from(v: Vec<f32>) -> Self {
        let (ptr, len) = shrinked_view(v);
        Self { ptr, len }
    }
}

impl TruckU8Array {
    pub(crate) fn from(v: Vec<u8>) -> Self {
        let (ptr, len) = shrinked_view(v);
        Self { ptr, len }
    }
}

impl TruckU32Array {
    pub(crate) fn from(v: Vec<u32>) -> Self {
        let (ptr, len) = shrinked_view(v);
        Self { ptr, len }
    }
}

impl TruckStr {
    pub(crate) fn from(v: Vec<u8>) -> Self {
        let (ptr, len) = shrinked_view(v);
        Self { ptr, len }
    }

    /// An empty view (NULL pointer, zero length) — safe to free.
    pub(crate) fn empty() -> Self {
        Self { ptr: std::ptr::null_mut(), len: 0 }
    }
}

// ---------------------------------------------------------------------------
// C ABI — array/string free functions
// ---------------------------------------------------------------------------

/// Free a `TruckF64Array` previously returned by truck-bridge. Idempotent for
/// an empty/zero-length array.
///
/// # Safety
/// `arr` must describe an allocation previously produced by truck-bridge (or be
/// the zero-initialized empty value), and must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_f64array_free(arr: TruckF64Array) {
    if arr.ptr.is_null() {
        return;
    }
    // SAFETY: produced by TruckF64Array::from with cap == len.
    drop(unsafe { vec_from_raw_parts::<f64>(arr.ptr, arr.len, arr.len) });
}

/// Free a `TruckF32Array`. Idempotent for an empty array. See
/// [`truck_f64array_free`] for the safety contract.
#[no_mangle]
pub unsafe extern "C" fn truck_f32array_free(arr: TruckF32Array) {
    if arr.ptr.is_null() {
        return;
    }
    drop(unsafe { vec_from_raw_parts::<f32>(arr.ptr, arr.len, arr.len) });
}

/// Free a `TruckU8Array`. Idempotent for an empty array. See
/// [`truck_f64array_free`] for the safety contract.
#[no_mangle]
pub unsafe extern "C" fn truck_u8array_free(arr: TruckU8Array) {
    if arr.ptr.is_null() {
        return;
    }
    drop(unsafe { vec_from_raw_parts::<u8>(arr.ptr, arr.len, arr.len) });
}

/// Free a `TruckU32Array`. Idempotent for an empty array. See
/// [`truck_f64array_free`] for the safety contract.
#[no_mangle]
pub unsafe extern "C" fn truck_u32array_free(arr: TruckU32Array) {
    if arr.ptr.is_null() {
        return;
    }
    drop(unsafe { vec_from_raw_parts::<u32>(arr.ptr, arr.len, arr.len) });
}

/// Free a `TruckStr`. Idempotent for an empty string. See
/// [`truck_f64array_free`] for the safety contract.
#[no_mangle]
pub unsafe extern "C" fn truck_str_free(s: TruckStr) {
    if s.ptr.is_null() {
        return;
    }
    drop(unsafe { vec_from_raw_parts::<u8>(s.ptr, s.len, s.len) });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_ref_null_is_none() {
        // SAFETY: NULL is explicitly allowed.
        assert!(unsafe { from_ref::<i32>(std::ptr::null()) }.is_none());
    }

    #[test]
    fn into_and_take_roundtrip() {
        let p = into_raw(123i32);
        // SAFETY: p came from into_raw, not yet reclaimed.
        assert_eq!(unsafe { take_raw(p) }, Some(123));
        // second take of the same address would be UB; we only check NULL case:
        assert!(unsafe { take_raw::<i32>(std::ptr::null_mut()) }.is_none());
    }

    #[test]
    fn f64array_from_and_free_preserves_data() {
        let arr = TruckF64Array::from(vec![1.5, -2.0, 3.25]);
        assert_eq!(arr.len, 3);
        // SAFETY: arr.ptr is valid for arr.len reads, just-produced.
        let slice = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        assert_eq!(slice, &[1.5, -2.0, 3.25]);
        // SAFETY: arr came from TruckF64Array::from.
        unsafe { truck_f64array_free(arr) };
    }

    #[test]
    fn f32array_from_and_free_preserves_data() {
        let arr = TruckF32Array::from(vec![1.5_f32, -2.0, 3.25]);
        assert_eq!(arr.len, 3);
        // SAFETY: arr.ptr is valid for arr.len reads, just-produced.
        let slice = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        assert_eq!(slice, &[1.5_f32, -2.0, 3.25]);
        // SAFETY: arr came from TruckF32Array::from.
        unsafe { truck_f32array_free(arr) };
    }

    #[test]
    fn u32array_from_and_free_preserves_data() {
        let arr = TruckU32Array::from(vec![1u32, 2, 3]);
        assert_eq!(arr.len, 3);
        // SAFETY: arr.ptr is valid for arr.len reads, just-produced.
        let slice = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        assert_eq!(slice, &[1u32, 2, 3]);
        // SAFETY: arr came from TruckU32Array::from.
        unsafe { truck_u32array_free(arr) };
    }

    #[test]
    fn empty_str_free_is_noop() {
        // SAFETY: empty value is explicitly allowed.
        unsafe { truck_str_free(TruckStr::empty()) };
    }

    #[test]
    fn str_from_bytes_roundtrip() {
        let s = TruckStr::from(b"hello".to_vec());
        assert_eq!(s.len, 5);
        // SAFETY: just produced, valid for s.len bytes.
        let bytes = unsafe { std::slice::from_raw_parts(s.ptr, s.len) };
        assert_eq!(bytes, b"hello");
        // SAFETY: came from TruckStr::from.
        unsafe { truck_str_free(s) };
    }
}

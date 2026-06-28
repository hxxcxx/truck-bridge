//! Topology layer C ABI surface — the B-rep types (`Vertex`, `Edge`, `Wire`,
//! `Face`, `Shell`, `Solid`).
//!
//! Stage 4a (current): `Vertex` only — construct, read back its point, free.
//! This is the **stress test** that truck-modeling's concrete topology types
//! cross the FFI boundary cleanly. Once it holds, `Edge`/`Face`/.../`Solid`
//! (stages 4b–4d) are the same pattern with a different inner type.
//!
//! All concrete topology types here come from `truck_modeling`, which already
//! monomorphizes `Vertex<P>` / `Edge<P, C>` / `Solid<P, C, S>` to
//! `<Point3, Curve, Surface>` via its `prelude!` macro. We never touch the
//! generic forms.
//!
//! ## Conventions (shared with the rest of the crate)
//!
//! - Opaque handle newtypes (no `#[repr(C)]`); C sees only `typedef struct`.
//! - Every `extern "C"` body runs under `truck_guard!` so panics never unwind.
//! - NULL inputs are rejected first; every `*_free` is idempotent / NULL-safe.
//! - No `let ... else` (cbindgen 0.x cannot parse it) — use `match`.

use crate::handle::{self, TruckF64Array};
use truck_modeling::{builder, Point3, Vertex};

/// Opaque handle to a truck `Vertex` (concrete `<Point3>` form). C sees only
/// `typedef struct TruckVertex TruckVertex;`.
///
/// A `Vertex` internally holds `Arc<Mutex<Point3>>`; it is `Send + Sync`, so
/// the handle may be moved between threads. Concurrent reads of its point are
/// fine; concurrent mutation is not — serialize such access in the caller.
#[derive(Debug)]
pub struct TruckVertex(pub(crate) Vertex);

/// Create a vertex from its `(x, y, z)` coordinates.
///
/// Returns an owning handle; release it with [`truck_vertex_free`].
#[no_mangle]
pub extern "C" fn truck_vertex_new(x: f64, y: f64, z: f64) -> *mut TruckVertex {
    handle::into_raw(TruckVertex(builder::vertex(Point3::new(x, y, z))))
}

/// Write the vertex's `(x, y, z)` coordinates into `out` as three `f64`s.
///
/// Returns `true` on success, `false` if `vertex` or `out` is NULL. The caller
/// owns the returned array and must free it with
/// [`crate::handle::truck_f64array_free`].
///
/// # Safety
/// `vertex` must be NULL or a valid handle; `out` must be a valid pointer to a
/// `TruckF64Array` that will be overwritten.
#[no_mangle]
pub unsafe extern "C" fn truck_vertex_point(
    vertex: *const TruckVertex,
    out: *mut TruckF64Array,
) -> bool {
    // SAFETY: caller guarantees vertex is NULL or a valid handle.
    let v = match unsafe { handle::from_ref(vertex) } {
        Some(v) => v,
        None => return false,
    };
    // SAFETY: caller guarantees out is a valid pointer or NULL.
    let out_ref = match unsafe { handle::from_mut(out) } {
        Some(o) => o,
        None => return false,
    };

    // `Vertex::point(&self) -> P` returns an owned Point3 (clone), so there is
    // no lifetime coupling to the vertex.
    let p = v.0.point();
    *out_ref = TruckF64Array::from(vec![p[0], p[1], p[2]]);
    true
}

/// Free a vertex handle. Idempotent: `truck_vertex_free(NULL)` is a no-op.
///
/// # Safety
/// `vertex` must be NULL or a handle previously returned by truck-bridge, and
/// must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_vertex_free(vertex: *mut TruckVertex) {
    // SAFETY: caller guarantees vertex is NULL or a valid, owned handle.
    match unsafe { handle::take_raw(vertex) } {
        Some(v) => drop(v),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_non_null() {
        let v = truck_vertex_new(1.0, 2.0, 3.0);
        assert!(!v.is_null());
        // SAFETY: v came from truck_vertex_new, not yet freed.
        unsafe { truck_vertex_free(v) };
    }

    #[test]
    fn point_roundtrip() {
        let v = truck_vertex_new(1.0, 2.0, 3.0);
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        // SAFETY: v is a fresh handle; arr is a valid out-pointer.
        let ok = unsafe { truck_vertex_point(v, &mut arr) };
        assert!(ok);
        assert_eq!(arr.len, 3);
        // SAFETY: arr.ptr valid for arr.len, just produced.
        let s = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        assert_eq!(s, &[1.0, 2.0, 3.0]);
        // SAFETY: arr came from truck_vertex_point.
        unsafe {
            crate::handle::truck_f64array_free(arr);
            truck_vertex_free(v);
        }
    }

    #[test]
    fn distinct_vertices_keep_distinct_points() {
        let a = truck_vertex_new(0.0, 0.0, 0.0);
        let b = truck_vertex_new(5.0, -2.0, 7.0);
        let mut pa = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        let mut pb = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        unsafe {
            truck_vertex_point(a, &mut pa);
            truck_vertex_point(b, &mut pb);
        }
        // SAFETY: both valid for len.
        let sa = unsafe { std::slice::from_raw_parts(pa.ptr, pa.len) };
        let sb = unsafe { std::slice::from_raw_parts(pb.ptr, pb.len) };
        assert_eq!(sa, &[0.0, 0.0, 0.0]);
        assert_eq!(sb, &[5.0, -2.0, 7.0]);
        unsafe {
            crate::handle::truck_f64array_free(pa);
            crate::handle::truck_f64array_free(pb);
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn null_arguments_return_false() {
        let v = truck_vertex_new(1.0, 2.0, 3.0);
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        // NULL vertex
        assert!(!unsafe { truck_vertex_point(std::ptr::null(), &mut arr) });
        // NULL out
        assert!(!unsafe { truck_vertex_point(v, std::ptr::null_mut()) });
        unsafe { truck_vertex_free(v) };
    }

    #[test]
    fn free_null_is_safe() {
        // SAFETY: NULL is explicitly allowed.
        unsafe { truck_vertex_free(std::ptr::null_mut()) };
    }
}

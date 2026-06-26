//! `PolygonMesh` C ABI surface.
//!
//! Stage 2 (current): minimal read-only surface — `new_empty`, `bounding_box`,
//! `free` — used to stress-test the four foundations against a real truck type.
//! The full surface (`from_obj` / `to_buffer` / ...) arrives in stage 3.

use crate::handle::{self, TruckF64Array};
use truck_polymesh::PolygonMesh;

/// Opaque handle to a truck `PolygonMesh` (monomorphized to its default
/// `<StandardVertex, StandardAttributes>` form — the only form truck itself
/// uses in practice). C sees only `typedef struct TruckPolygonMesh ...;`.
#[derive(Debug)]
pub struct TruckPolygonMesh(pub(crate) PolygonMesh);

/// Create an empty polygon mesh (no positions, no faces).
///
/// Returns an owning handle; release it with [`truck_polygonmesh_free`].
#[no_mangle]
pub extern "C" fn truck_polygonmesh_new_empty() -> *mut TruckPolygonMesh {
    handle::into_raw(TruckPolygonMesh(PolygonMesh::default()))
}

/// Compute the axis-aligned bounding box and write it into `out` as six `f64`s:
/// `[min_x, min_y, min_z, max_x, max_y, max_z]`.
///
/// Returns `true` on success, `false` if `mesh` or `out` is NULL.
///
/// For an empty mesh truck reports an inverted box (`min = +INF`,
/// `max = -INF`); those values are passed through unchanged so consumers can
/// detect emptiness.
///
/// The caller owns the returned array and must free it with
/// [`crate::handle::truck_f64array_free`].
///
/// # Safety
/// `mesh` must be NULL or a valid handle; `out` must be a valid pointer to a
/// `TruckF64Array` that will be overwritten.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_bounding_box(
    mesh: *const TruckPolygonMesh,
    out: *mut TruckF64Array,
) -> bool {
    // SAFETY: caller guarantees mesh is NULL or a valid handle.
    let m = match unsafe { handle::from_ref(mesh) } {
        Some(m) => m,
        None => return false,
    };
    // SAFETY: caller guarantees out is a valid pointer or NULL.
    let out_ref = match unsafe { handle::from_mut(out) } {
        Some(o) => o,
        None => return false,
    };

    let bdd = m.0.bounding_box();
    let min = bdd.min();
    let max = bdd.max();
    let data = vec![min[0], min[1], min[2], max[0], max[1], max[2]];
    *out_ref = TruckF64Array::from(data);
    true
}

/// Free a polygon mesh handle. Idempotent: `truck_polygonmesh_free(NULL)` is a
/// no-op.
///
/// # Safety
/// `mesh` must be NULL or a handle previously returned by truck-bridge, and
/// must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_free(mesh: *mut TruckPolygonMesh) {
    // SAFETY: caller guarantees mesh is NULL or a valid, owned handle.
    match unsafe { handle::take_raw(mesh) } {
        Some(m) => drop(m),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_empty_is_non_null() {
        let m = truck_polygonmesh_new_empty();
        assert!(!m.is_null());
        // SAFETY: m came from new_empty, not yet freed.
        unsafe { truck_polygonmesh_free(m) };
    }

    #[test]
    fn empty_mesh_bounding_box_is_inverted_infinities() {
        let m = truck_polygonmesh_new_empty();
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        // SAFETY: m is a fresh handle; arr is a valid out-pointer.
        let ok = unsafe { truck_polygonmesh_bounding_box(m, &mut arr) };
        assert!(ok);
        assert_eq!(arr.len, 6);
        // SAFETY: arr.ptr valid for arr.len, just produced.
        let s = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        // empty box: min = +INF, max = -INF
        assert!(s[0].is_infinite() && s[0].is_sign_positive());
        assert!(s[3].is_infinite() && s[3].is_sign_negative());
        // SAFETY: arr came from bounding_box.
        unsafe { crate::handle::truck_f64array_free(arr) };
        unsafe { truck_polygonmesh_free(m) };
    }

    #[test]
    fn null_arguments_return_false() {
        let m = truck_polygonmesh_new_empty();
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        // NULL mesh
        assert!(!unsafe { truck_polygonmesh_bounding_box(std::ptr::null(), &mut arr) });
        // NULL out
        assert!(!unsafe { truck_polygonmesh_bounding_box(m, std::ptr::null_mut()) });
        unsafe { truck_polygonmesh_free(m) };
    }

    #[test]
    fn free_null_is_safe() {
        // SAFETY: NULL is explicitly allowed.
        unsafe { truck_polygonmesh_free(std::ptr::null_mut()) };
    }
}

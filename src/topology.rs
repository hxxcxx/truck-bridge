//! Topology layer C ABI surface — the B-rep types (`Vertex`, `Edge`, `Wire`,
//! `Face`, `Shell`, `Solid`).
//!
//! - Stage 4a: `Vertex` — construct, read back its point, free.
//! - Stage 4b: `Edge` — `line` / `circle_arc` (by transit) / `bezier`
//!   constructors, `front_vertex` / `back_vertex` queries, free.
//! - Stage 4c (current): `Face` — `homotopy` constructor, boundary edge
//!   count + enumeration (via the new `TruckEdgeArray` handle-array type),
//!   free.
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
use truck_modeling::{builder, Edge, Face, Point3, Vertex};

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

// ===========================================================================
// Stage 4b — Edge
// ===========================================================================

/// Opaque handle to a truck `Edge` (concrete `<Point3, Curve>` form). C sees
/// only `typedef struct TruckEdge TruckEdge;`.
#[derive(Debug)]
pub struct TruckEdge(pub(crate) Edge);

/// Read three `f64`s (`[x, y, z]`) from a C array, or `None` if `p` is NULL or
/// too short.
///
/// # Safety
/// `p` must be NULL or valid for `len` `f64`s.
unsafe fn read_vec3(p: *const f64, len: usize) -> Option<[f64; 3]> {
    if p.is_null() || len < 3 {
        return None;
    }
    // SAFETY: caller guarantees p is valid for len >= 3 f64s.
    let s = unsafe { std::slice::from_raw_parts(p, 3) };
    Some([s[0], s[1], s[2]])
}

/// Create a straight edge (line) between two vertices. The two input vertex
/// handles are **borrowed**, not consumed.
///
/// Returns a new edge handle, or NULL if either vertex is NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_line(
    v0: *const TruckVertex,
    v1: *const TruckVertex,
) -> *mut TruckEdge {
    // SAFETY: caller guarantees the handles are NULL or valid.
    let (a, b) = match (unsafe { handle::from_ref(v0) }, unsafe { handle::from_ref(v1) }) {
        (Some(a), Some(b)) => (a, b),
        _ => return std::ptr::null_mut(),
    };
    let res = crate::error::truck_guard!(|| Ok::<Edge, crate::error::TruckError>(builder::line(&a.0, &b.0)));
    match res {
        Ok(edge) => handle::into_raw(TruckEdge(edge)),
        Err(_panic) => std::ptr::null_mut(),
    }
}

/// Create a circular-arc edge from `v0` to `v1` that passes through `transit`
/// (`[x, y, z]`). The vertices are borrowed.
///
/// Returns a new edge handle, or NULL if `v0`/`v1`/`transit` is NULL/short or
/// the three points are degenerate (collinear), in which case truck panics
/// internally and the guard converts that to NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_circle_arc_by_transit(
    v0: *const TruckVertex,
    v1: *const TruckVertex,
    transit: *const f64,
    transit_len: usize,
) -> *mut TruckEdge {
    // SAFETY: caller guarantees the handles are NULL or valid.
    let (a, b) = match (unsafe { handle::from_ref(v0) }, unsafe { handle::from_ref(v1) }) {
        (Some(a), Some(b)) => (a, b),
        _ => return std::ptr::null_mut(),
    };
    // SAFETY: caller guarantees transit is NULL or valid for transit_len f64s.
    let [x, y, z] = match unsafe { read_vec3(transit, transit_len) } {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let res = crate::error::truck_guard!(|| {
        Ok::<Edge, crate::error::TruckError>(builder::circle_arc(&a.0, &b.0, Point3::new(x, y, z)))
    });
    match res {
        Ok(edge) => handle::into_raw(TruckEdge(edge)),
        Err(_panic) => std::ptr::null_mut(),
    }
}

/// Create a Bezier-curve edge from `v0` to `v1` with intermediate control
/// points. `ctrl` is a flat array of `ctrl_len` `f64`s, i.e. `ctrl_len / 3`
/// control points, each `[x, y, z]`. `ctrl_len` must be a multiple of 3
/// (`ctrl` may be NULL with `ctrl_len == 0` for a straight segment).
///
/// The vertices are borrowed.
///
/// Returns a new edge handle, or NULL on NULL vertices or a non-multiple-of-3
/// `ctrl_len`.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_bezier(
    v0: *const TruckVertex,
    v1: *const TruckVertex,
    ctrl: *const f64,
    ctrl_len: usize,
) -> *mut TruckEdge {
    let (a, b) = match (unsafe { handle::from_ref(v0) }, unsafe { handle::from_ref(v1) }) {
        (Some(a), Some(b)) => (a, b),
        _ => return std::ptr::null_mut(),
    };
    if ctrl_len % 3 != 0 {
        return std::ptr::null_mut();
    }
    let points: Vec<Point3> = if ctrl.is_null() || ctrl_len == 0 {
        Vec::new()
    } else {
        // SAFETY: caller guarantees ctrl is valid for ctrl_len f64s.
        let s = unsafe { std::slice::from_raw_parts(ctrl, ctrl_len) };
        s.chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect()
    };
    let res = crate::error::truck_guard!(|| {
        Ok::<Edge, crate::error::TruckError>(builder::bezier(&a.0, &b.0, points))
    });
    match res {
        Ok(edge) => handle::into_raw(TruckEdge(edge)),
        Err(_panic) => std::ptr::null_mut(),
    }
}

/// Return the front (start) vertex of `edge` as a **new, independent** handle.
///
/// The returned `TruckVertex` is a clone and must be freed separately with
/// [`truck_vertex_free`]; it is decoupled from `edge`'s lifetime.
///
/// Returns NULL if `edge` is NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_front_vertex(edge: *const TruckEdge) -> *mut TruckVertex {
    // SAFETY: caller guarantees edge is NULL or a valid handle.
    let e = match unsafe { handle::from_ref(edge) } {
        Some(e) => e,
        None => return std::ptr::null_mut(),
    };
    // front() borrows; clone to get an owned Vertex we can hand across FFI.
    handle::into_raw(TruckVertex(e.0.front().clone()))
}

/// Return the back (end) vertex of `edge` as a **new, independent** handle.
///
/// See [`truck_edge_front_vertex`]: the result is a clone with its own lifetime.
/// Returns NULL if `edge` is NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_back_vertex(edge: *const TruckEdge) -> *mut TruckVertex {
    // SAFETY: caller guarantees edge is NULL or a valid handle.
    let e = match unsafe { handle::from_ref(edge) } {
        Some(e) => e,
        None => return std::ptr::null_mut(),
    };
    handle::into_raw(TruckVertex(e.0.back().clone()))
}

/// Free an edge handle. Idempotent: `truck_edge_free(NULL)` is a no-op.
///
/// # Safety
/// `edge` must be NULL or a handle previously returned by truck-bridge, and
/// must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_free(edge: *mut TruckEdge) {
    // SAFETY: caller guarantees edge is NULL or a valid, owned handle.
    match unsafe { handle::take_raw(edge) } {
        Some(e) => drop(e),
        None => {}
    }
}

// ===========================================================================
// Stage 4c — Face + TruckEdgeArray
// ===========================================================================

/// Opaque handle to a truck `Face` (concrete `<Point3, Curve, Surface>` form).
/// C sees only `typedef struct TruckFace TruckFace;`.
#[derive(Debug)]
pub struct TruckFace(pub(crate) Face);

/// Owned array of opaque edge handles (`*mut *mut TruckEdge`).
///
/// **Two-layer ownership** (read carefully):
///   - The array container itself is one allocation (`ptr[0..len]` of edge
///     handle pointers). Free it with [`truck_edgearray_free`].
///   - Each element `ptr[i]` is an **independent** edge handle that must be
///     released separately with [`truck_edge_free`].
///
/// For the common case where you want to release everything at once, use
/// [`truck_edgearray_free_all`], which frees both the container and every
/// handle inside it.
#[repr(C)]
#[derive(Debug)]
pub struct TruckEdgeArray {
    /// Pointer to `len` edge handles (`*mut TruckEdge`).
    pub ptr: *mut *mut TruckEdge,
    /// Number of handles.
    pub len: usize,
}

impl TruckEdgeArray {
    /// An empty array (NULL pointer, zero length) — safe to free either way.
    /// Kept for callers/tests that want a zero-initialized starting value.
    #[allow(dead_code)]
    fn empty() -> Self {
        Self { ptr: std::ptr::null_mut(), len: 0 }
    }
}

/// Create a homotopic face sweeping from `e0` to `e1`. The two edge handles
/// are **borrowed**, not consumed.
///
/// Returns a new face handle, or NULL if either edge is NULL or the edges are
/// geometrically incompatible (truck panics internally; the guard converts
/// that to NULL).
#[no_mangle]
pub unsafe extern "C" fn truck_face_homotopy(
    e0: *const TruckEdge,
    e1: *const TruckEdge,
) -> *mut TruckFace {
    // SAFETY: caller guarantees the handles are NULL or valid.
    let (a, b) = match (unsafe { handle::from_ref(e0) }, unsafe { handle::from_ref(e1) }) {
        (Some(a), Some(b)) => (a, b),
        _ => return std::ptr::null_mut(),
    };
    let res = crate::error::truck_guard!(|| {
        Ok::<Face, crate::error::TruckError>(builder::homotopy(&a.0, &b.0))
    });
    match res {
        Ok(face) => handle::into_raw(TruckFace(face)),
        Err(_panic) => std::ptr::null_mut(),
    }
}

/// Count the boundary edges of `face`.
///
/// Returns 0 if `face` is NULL. `homotopy` produces exactly 4 boundary edges.
#[no_mangle]
pub unsafe extern "C" fn truck_face_boundary_edge_count(face: *const TruckFace) -> usize {
    // SAFETY: caller guarantees face is NULL or a valid handle.
    let f = match unsafe { handle::from_ref(face) } {
        Some(f) => f,
        None => return 0,
    };
    f.0.boundary_iters().into_iter().flatten().count()
}

/// Enumerate the boundary edges of `face` as a [`TruckEdgeArray`] of
/// **independent** edge handles.
///
/// Each returned edge is owned by the caller; free the whole result with
/// [`truck_edgearray_free_all`] (or free each handle with [`truck_edge_free`]
/// and the container with [`truck_edgearray_free`]).
///
/// Returns `true` on success, `false` if `face` or `out` is NULL.
///
/// # Safety
/// `face` must be NULL or a valid handle; `out` must be a valid pointer to a
/// `TruckEdgeArray` that will be overwritten.
#[no_mangle]
pub unsafe extern "C" fn truck_face_boundary_edges(
    face: *const TruckFace,
    out: *mut TruckEdgeArray,
) -> bool {
    // SAFETY: caller guarantees face is NULL or a valid handle.
    let f = match unsafe { handle::from_ref(face) } {
        Some(f) => f,
        None => return false,
    };
    // SAFETY: caller guarantees out is a valid pointer or NULL.
    let out_ref = match unsafe { handle::from_mut(out) } {
        Some(o) => o,
        None => return false,
    };

    // boundary_iters().flatten() yields owned Edge values; wrap each into a
    // fresh TruckEdge handle.
    let handles: Vec<*mut TruckEdge> = f
        .0
        .boundary_iters()
        .into_iter()
        .flatten()
        .map(|edge| handle::into_raw(TruckEdge(edge)))
        .collect();
    let len = handles.len();
    let (ptr, _len, _cap) = handle::vec_into_raw_parts(handles);
    *out_ref = TruckEdgeArray { ptr, len };
    true
}

/// Free a face handle. Idempotent: `truck_face_free(NULL)` is a no-op.
///
/// # Safety
/// `face` must be NULL or a handle previously returned by truck-bridge, and
/// must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_face_free(face: *mut TruckFace) {
    // SAFETY: caller guarantees face is NULL or a valid, owned handle.
    match unsafe { handle::take_raw(face) } {
        Some(f) => drop(f),
        None => {}
    }
}

/// Free only the `TruckEdgeArray` **container** (the pointer array), leaving
/// the individual edge handles untouched. Use this when you intend to keep the
/// edges; otherwise prefer [`truck_edgearray_free_all`]. Idempotent for an
/// empty array.
///
/// # Safety
/// `arr` must describe a container previously produced by truck-bridge (or be
/// the empty value), and must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_edgearray_free(arr: TruckEdgeArray) {
    if arr.ptr.is_null() {
        return;
    }
    // SAFETY: arr.ptr came from vec_into_raw_parts on Vec<*mut TruckEdge>, with
    // cap == len (shrink_to_fit in Vec::collect does not guarantee cap==len, so
    // we recover using len for both — the handles themselves are separate Box
    // allocations and are NOT dropped here by design).
    drop(unsafe {
        handle::vec_from_raw_parts::<*mut TruckEdge>(arr.ptr, arr.len, arr.len)
    });
}

/// Free the `TruckEdgeArray` **container and every edge handle inside it**.
/// This is the common choice. Idempotent for an empty array.
///
/// # Safety
/// `arr` must describe a container + handles previously produced by
/// truck-bridge, none already freed. After this call, none of the handles nor
/// the container may be used again.
#[no_mangle]
pub unsafe extern "C" fn truck_edgearray_free_all(arr: TruckEdgeArray) {
    if arr.ptr.is_null() {
        return;
    }
    // SAFETY: arr.ptr valid for arr.len handle pointers.
    let slice = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
    for &h in slice {
        // SAFETY: each h is an independent, owned TruckEdge handle (or NULL,
        // which truck_edge_free tolerates).
        unsafe { truck_edge_free(h) };
    }
    // SAFETY: same as truck_edgearray_free — recover the container Vec.
    drop(unsafe {
        handle::vec_from_raw_parts::<*mut TruckEdge>(arr.ptr, arr.len, arr.len)
    });
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

    // ---- stage 4b: Edge ----------------------------------------------------

    /// Helper: read a vertex's point into a Vec<f64>, freeing the array.
    unsafe fn vertex_point_vec(v: *const TruckVertex) -> Vec<f64> {
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        // SAFETY: v is a valid handle; arr is valid.
        unsafe { truck_vertex_point(v, &mut arr) };
        // SAFETY: arr.ptr valid for arr.len.
        let s = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) }.to_vec();
        unsafe { crate::handle::truck_f64array_free(arr) };
        s
    }

    #[test]
    fn edge_line_endpoints_match() {
        let a = truck_vertex_new(1.0, 2.0, 3.0);
        let b = truck_vertex_new(4.0, 5.0, 6.0);
        // SAFETY: a, b are valid handles.
        let e = unsafe { truck_edge_line(a, b) };
        assert!(!e.is_null(), "line edge should be non-null");
        // front should equal a, back should equal b.
        // SAFETY: e valid.
        let f = unsafe { truck_edge_front_vertex(e) };
        let bk = unsafe { truck_edge_back_vertex(e) };
        assert!(!f.is_null() && !bk.is_null());
        assert_eq!(unsafe { vertex_point_vec(f) }, vec![1.0, 2.0, 3.0]);
        assert_eq!(unsafe { vertex_point_vec(bk) }, vec![4.0, 5.0, 6.0]);
        unsafe {
            truck_vertex_free(f);
            truck_vertex_free(bk);
            truck_edge_free(e);
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn edge_line_null_inputs() {
        let b = truck_vertex_new(0.0, 0.0, 0.0);
        // SAFETY: first arg NULL.
        assert!(unsafe { truck_edge_line(std::ptr::null(), b) }.is_null());
        // SAFETY: second arg NULL.
        assert!(unsafe { truck_edge_line(b, std::ptr::null()) }.is_null());
        unsafe { truck_vertex_free(b) };
    }

    #[test]
    fn edge_circle_arc_by_transit_constructs() {
        let a = truck_vertex_new(1.0, 0.0, 0.0);
        let b = truck_vertex_new(-1.0, 0.0, 0.0);
        let transit = [0.0, 1.0, 0.0]; // upper semicircle
        // SAFETY: a, b, transit valid.
        let e = unsafe { truck_edge_circle_arc_by_transit(a, b, transit.as_ptr(), 3) };
        assert!(!e.is_null(), "circle arc should be non-null");
        unsafe {
            truck_edge_free(e);
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn edge_circle_arc_null_transit_returns_null() {
        let a = truck_vertex_new(1.0, 0.0, 0.0);
        let b = truck_vertex_new(-1.0, 0.0, 0.0);
        // SAFETY: NULL transit pointer.
        let e = unsafe { truck_edge_circle_arc_by_transit(a, b, std::ptr::null(), 0) };
        assert!(e.is_null());
        unsafe {
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn edge_bezier_constructs() {
        let a = truck_vertex_new(0.0, 0.0, 0.0);
        let b = truck_vertex_new(3.0, 0.0, 0.0);
        // two intermediate control points
        let ctrl = [1.0, 1.0, 0.0, 2.0, 1.0, 0.0];
        // SAFETY: a, b, ctrl valid.
        let e = unsafe { truck_edge_bezier(a, b, ctrl.as_ptr(), ctrl.len()) };
        assert!(!e.is_null(), "bezier edge should be non-null");
        unsafe {
            truck_edge_free(e);
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn edge_bezier_bad_length_returns_null() {
        let a = truck_vertex_new(0.0, 0.0, 0.0);
        let b = truck_vertex_new(1.0, 0.0, 0.0);
        let bad = [1.0, 2.0]; // length 2, not a multiple of 3
        // SAFETY: a, b valid; bad is valid for 2.
        let e = unsafe { truck_edge_bezier(a, b, bad.as_ptr(), bad.len()) };
        assert!(e.is_null(), "non-multiple-of-3 ctrl length should yield NULL");
        unsafe {
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn edge_front_back_survive_edge_free() {
        // The cloned front/back handles must remain valid after the edge is freed.
        let a = truck_vertex_new(1.0, 2.0, 3.0);
        let b = truck_vertex_new(4.0, 5.0, 6.0);
        // SAFETY: a, b valid.
        let e = unsafe { truck_edge_line(a, b) };
        let f = unsafe { truck_edge_front_vertex(e) };
        unsafe { truck_edge_free(e) };
        // f is an independent clone; querying its point must still work.
        assert_eq!(unsafe { vertex_point_vec(f) }, vec![1.0, 2.0, 3.0]);
        unsafe {
            truck_vertex_free(f);
            truck_vertex_free(a);
            truck_vertex_free(b);
        }
    }

    #[test]
    fn edge_free_null_is_safe() {
        // SAFETY: NULL is explicitly allowed.
        unsafe { truck_edge_free(std::ptr::null_mut()) };
    }

    // ---- stage 4c: Face ----------------------------------------------------

    /// Helper: build two edges sharing the homotopy shape used by tests.
    /// Returns (v0, v1, v2, v3, edge0, edge1).
    fn homotopy_edges() -> (
        *mut TruckVertex,
        *mut TruckVertex,
        *mut TruckVertex,
        *mut TruckVertex,
        *mut TruckEdge,
        *mut TruckEdge,
    ) {
        let v0 = truck_vertex_new(0.0, 0.0, 0.0);
        let v1 = truck_vertex_new(1.0, 0.0, 0.0);
        let v2 = truck_vertex_new(0.0, 0.0, 1.0);
        let v3 = truck_vertex_new(1.0, 0.0, 1.0);
        // SAFETY: all vertices valid.
        let e0 = unsafe { truck_edge_line(v0, v1) };
        let e1 = unsafe { truck_edge_line(v2, v3) };
        (v0, v1, v2, v3, e0, e1)
    }

    #[test]
    fn face_homotopy_constructs() {
        let (v0, v1, v2, v3, e0, e1) = homotopy_edges();
        // SAFETY: e0, e1 valid.
        let face = unsafe { truck_face_homotopy(e0, e1) };
        assert!(!face.is_null(), "homotopy face should be non-null");
        // homotopy produces exactly 4 boundary edges.
        // SAFETY: face valid.
        let count = unsafe { truck_face_boundary_edge_count(face) };
        assert_eq!(count, 4, "homotopy face should have 4 boundary edges");
        unsafe {
            truck_face_free(face);
            truck_edge_free(e0);
            truck_edge_free(e1);
            truck_vertex_free(v0);
            truck_vertex_free(v1);
            truck_vertex_free(v2);
            truck_vertex_free(v3);
        }
    }

    #[test]
    fn face_homotopy_null_inputs() {
        let (_v0, _v1, _v2, _v3, e0, e1) = homotopy_edges();
        // SAFETY: first arg NULL.
        assert!(unsafe { truck_face_homotopy(std::ptr::null(), e1) }.is_null());
        // SAFETY: second arg NULL.
        assert!(unsafe { truck_face_homotopy(e0, std::ptr::null()) }.is_null());
        unsafe {
            truck_edge_free(e0);
            truck_edge_free(e1);
            truck_vertex_free(_v0);
            truck_vertex_free(_v1);
            truck_vertex_free(_v2);
            truck_vertex_free(_v3);
        }
    }

    #[test]
    fn face_boundary_edges_array() {
        let (v0, v1, v2, v3, e0, e1) = homotopy_edges();
        // SAFETY: e0, e1 valid.
        let face = unsafe { truck_face_homotopy(e0, e1) };
        let mut arr = TruckEdgeArray { ptr: std::ptr::null_mut(), len: 0 };
        // SAFETY: face valid; arr valid out-pointer.
        let ok = unsafe { truck_face_boundary_edges(face, &mut arr) };
        assert!(ok);
        assert_eq!(arr.len, 4);
        // each handle non-null
        // SAFETY: arr.ptr valid for arr.len.
        let slice = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        for &h in slice {
            assert!(!h.is_null(), "boundary edge handle must be non-null");
        }
        // free_all releases container + all handles
        // SAFETY: arr produced by boundary_edges.
        unsafe { truck_edgearray_free_all(arr) };
        unsafe {
            truck_face_free(face);
            truck_edge_free(e0);
            truck_edge_free(e1);
            truck_vertex_free(v0);
            truck_vertex_free(v1);
            truck_vertex_free(v2);
            truck_vertex_free(v3);
        }
    }

    #[test]
    fn face_boundary_edges_null_face_returns_false() {
        let mut arr = TruckEdgeArray { ptr: std::ptr::null_mut(), len: 0 };
        // SAFETY: NULL face explicitly handled.
        assert!(!unsafe { truck_face_boundary_edges(std::ptr::null(), &mut arr) });
    }

    #[test]
    fn face_free_null_is_safe() {
        // SAFETY: NULL is explicitly allowed.
        unsafe { truck_face_free(std::ptr::null_mut()) };
    }

    #[test]
    fn edgearray_free_all_empty_is_safe() {
        // SAFETY: empty value is explicitly allowed.
        unsafe { truck_edgearray_free_all(TruckEdgeArray::empty()) };
        // SAFETY: empty value is explicitly allowed.
        unsafe { truck_edgearray_free(TruckEdgeArray::empty()) };
    }
}

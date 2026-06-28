//! Topology layer C ABI surface — the B-rep types (`Vertex`, `Edge`, `Wire`,
//! `Face`, `Shell`, `Solid`).
//!
//! - Stage 4a: `Vertex` — construct, read back its point, free.
//! - Stage 4b: `Edge` — `line` / `circle_arc` (by transit) / `bezier`
//!   constructors, `front_vertex` / `back_vertex` queries, free.
//! - Stage 4c: `Face` — `homotopy` constructor, boundary edge count +
//!   enumeration (via the `TruckEdgeArray` handle-array type), free.
//! - Stage 4d (current): `Shell` / `Solid` — construct from sub-shapes,
//!   tessellate to `PolygonMesh`; `AbstractShape` polymorphic handle with
//!   `tsweep` / `rsweep` / `translated` / `rotated` / `scaled`.
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

use crate::error::TruckError;
use crate::handle::{self, TruckF64Array};
use crate::polymesh::TruckPolygonMesh;
use truck_meshalgo::tessellation::{MeshedShape, RobustMeshableShape};
use truck_modeling::{builder, Edge, Face, Point3, Shell, Solid, Vector3, Vertex};

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

// ===========================================================================
// Stage 4d — Shell / Solid / AbstractShape / sweep / transform
// ===========================================================================

/// Opaque handle to a truck `Shell` (concrete `<Point3, Curve, Surface>` form).
#[derive(Debug)]
pub struct TruckShell(pub(crate) Shell);

/// Opaque handle to a truck `Solid` (concrete `<Point3, Curve, Surface>` form).
#[derive(Debug)]
pub struct TruckSolid(pub(crate) Solid);

/// Polymorphic topology handle — wraps any concrete topology type so that
/// `tsweep` / `rsweep` / transform operations can accept "any shape" and
/// return "any shape" (the result type depends on the input type).
///
/// Build one with `truck_{vertex,edge,face,shell,solid}_upcast`, inspect with
/// `truck_abstractshape_is_*`, extract with `truck_abstractshape_into_*`
/// (which consumes the `AbstractShape`).
#[derive(Debug)]
pub struct AbstractShape(pub(crate) SubShape);

#[derive(Debug)]
pub(crate) enum SubShape {
    Vertex(TruckVertex),
    Edge(TruckEdge),
    Face(TruckFace),
    Shell(TruckShell),
    Solid(TruckSolid),
}

impl AbstractShape {
    pub(crate) fn from_vertex(v: TruckVertex) -> Self {
        AbstractShape(SubShape::Vertex(v))
    }
    pub(crate) fn from_edge(e: TruckEdge) -> Self {
        AbstractShape(SubShape::Edge(e))
    }
    pub(crate) fn from_face(f: TruckFace) -> Self {
        AbstractShape(SubShape::Face(f))
    }
    pub(crate) fn from_shell(s: TruckShell) -> Self {
        AbstractShape(SubShape::Shell(s))
    }
    pub(crate) fn from_solid(s: TruckSolid) -> Self {
        AbstractShape(SubShape::Solid(s))
    }
}

// ---------------------------------------------------------------------------
// Shell
// ---------------------------------------------------------------------------

/// Build a shell from an array of face handles. **The face handles are
/// consumed** (moved into the shell) and must not be used or freed afterwards.
///
/// Returns a new shell handle on success, or NULL if `faces` is NULL or `count`
/// is 0.
///
/// # Safety
/// `faces` must be NULL or a valid array of `count` `*mut TruckFace` handles,
/// each a valid, owned handle (none already freed/consumed).
#[no_mangle]
pub unsafe extern "C" fn truck_shell_from_faces(
    faces: *const *mut TruckFace,
    count: usize,
) -> *mut TruckShell {
    if faces.is_null() || count == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees faces is valid for count handle pointers.
    let slice = unsafe { std::slice::from_raw_parts(faces, count) };
    let mut vec = Vec::with_capacity(count);
    for &h in slice {
        // SAFETY: each h is an owned TruckFace handle being consumed here.
        match unsafe { handle::take_raw(h) } {
            Some(f) => vec.push(f.0),
            None => return std::ptr::null_mut(),
        }
    }
    handle::into_raw(TruckShell(Shell::from(vec)))
}

/// Free a shell handle. Idempotent.
///
/// # Safety
/// `shell` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_shell_free(shell: *mut TruckShell) {
    match unsafe { handle::take_raw(shell) } {
        Some(s) => drop(s),
        None => {}
    }
}

/// Tessellate a shell into a `PolygonMesh` at tolerance `tol`.
///
/// On success writes a new mesh handle to `*out_mesh`; on failure (NULL shell,
/// degenerate geometry) writes an error handle to `*err` and returns false.
///
/// # Safety
/// `shell` must be NULL or a valid handle; `out_mesh`/`err` valid or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_shell_to_polygon(
    shell: *const TruckShell,
    tol: f64,
    out_mesh: *mut *mut TruckPolygonMesh,
    err: *mut *mut TruckError,
) -> bool {
    if out_mesh.is_null() {
        return false;
    }
    // SAFETY: caller guarantees shell is NULL or valid.
    let s = match unsafe { handle::from_ref(shell) } {
        Some(s) => s,
        None => return false,
    };
    let res = crate::error::truck_guard!(|| {
        if tol <= 0.0 {
            return Err(TruckError::new(format!("tolerance must be positive, got {tol}")));
        }
        let meshed = s.0.robust_triangulation(tol);
        let polygon: truck_polymesh::PolygonMesh = meshed.to_polygon();
        Ok::<_, TruckError>(polygon)
    });
    crate::truck_deliver!(res, err, |m: truck_polymesh::PolygonMesh| {
        // SAFETY: out_mesh checked non-NULL above.
        unsafe { *out_mesh = handle::into_raw(TruckPolygonMesh(m)) };
    })
}

// ---------------------------------------------------------------------------
// Solid
// ---------------------------------------------------------------------------

/// Build a solid from an array of shell handles. **The shell handles are
/// consumed** (moved into the solid) and must not be used or freed afterwards.
///
/// Returns a new solid handle, or NULL if `shells` is NULL or `count` is 0.
///
/// # Safety
/// `shells` must be NULL or a valid array of `count` owned shell handles.
#[no_mangle]
pub unsafe extern "C" fn truck_solid_from_shells(
    shells: *const *mut TruckShell,
    count: usize,
) -> *mut TruckSolid {
    if shells.is_null() || count == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: caller guarantees shells is valid for count handle pointers.
    let slice = unsafe { std::slice::from_raw_parts(shells, count) };
    let mut vec = Vec::with_capacity(count);
    for &h in slice {
        match unsafe { handle::take_raw(h) } {
            Some(s) => vec.push(s.0),
            None => return std::ptr::null_mut(),
        }
    }
    handle::into_raw(TruckSolid(Solid::new(vec)))
}

/// Free a solid handle. Idempotent.
///
/// # Safety
/// `solid` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_solid_free(solid: *mut TruckSolid) {
    match unsafe { handle::take_raw(solid) } {
        Some(s) => drop(s),
        None => {}
    }
}

/// Tessellate a solid into a `PolygonMesh` at tolerance `tol` (uses the first
/// boundary shell). See [`truck_shell_to_polygon`] for the error contract.
///
/// # Safety
/// `solid` must be NULL or valid; `out_mesh`/`err` valid or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_solid_to_polygon(
    solid: *const TruckSolid,
    tol: f64,
    out_mesh: *mut *mut TruckPolygonMesh,
    err: *mut *mut TruckError,
) -> bool {
    if out_mesh.is_null() {
        return false;
    }
    // SAFETY: caller guarantees solid is NULL or valid.
    let s = match unsafe { handle::from_ref(solid) } {
        Some(s) => s,
        None => return false,
    };
    let res = crate::error::truck_guard!(|| {
        if tol <= 0.0 {
            return Err(TruckError::new(format!("tolerance must be positive, got {tol}")));
        }
        if s.0.boundaries().is_empty() {
            return Err(TruckError::new("solid has no boundary shells"));
        }
        let meshed = s.0.robust_triangulation(tol);
        let shell = &meshed.boundaries()[0];
        let polygon: truck_polymesh::PolygonMesh = shell.to_polygon();
        Ok::<_, TruckError>(polygon)
    });
    crate::truck_deliver!(res, err, |m: truck_polymesh::PolygonMesh| {
        // SAFETY: out_mesh checked non-NULL above.
        unsafe { *out_mesh = handle::into_raw(TruckPolygonMesh(m)) };
    })
}

// ---------------------------------------------------------------------------
// AbstractShape — upcast / inspect / downcast
// ---------------------------------------------------------------------------

/// Wrap a vertex into an `AbstractShape`. The vertex handle is consumed.
///
/// # Safety
/// `v` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_vertex_upcast(v: *mut TruckVertex) -> *mut AbstractShape {
    match unsafe { handle::take_raw(v) } {
        Some(v) => handle::into_raw(AbstractShape::from_vertex(v)),
        None => std::ptr::null_mut(),
    }
}

/// Wrap an edge into an `AbstractShape`. The edge handle is consumed.
///
/// # Safety
/// `e` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_edge_upcast(e: *mut TruckEdge) -> *mut AbstractShape {
    match unsafe { handle::take_raw(e) } {
        Some(e) => handle::into_raw(AbstractShape::from_edge(e)),
        None => std::ptr::null_mut(),
    }
}

/// Wrap a face into an `AbstractShape`. The face handle is consumed.
///
/// # Safety
/// `f` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_face_upcast(f: *mut TruckFace) -> *mut AbstractShape {
    match unsafe { handle::take_raw(f) } {
        Some(f) => handle::into_raw(AbstractShape::from_face(f)),
        None => std::ptr::null_mut(),
    }
}

/// Wrap a shell into an `AbstractShape`. The shell handle is consumed.
///
/// # Safety
/// `s` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_shell_upcast(s: *mut TruckShell) -> *mut AbstractShape {
    match unsafe { handle::take_raw(s) } {
        Some(s) => handle::into_raw(AbstractShape::from_shell(s)),
        None => std::ptr::null_mut(),
    }
}

/// Wrap a solid into an `AbstractShape`. The solid handle is consumed.
///
/// # Safety
/// `s` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_solid_upcast(s: *mut TruckSolid) -> *mut AbstractShape {
    match unsafe { handle::take_raw(s) } {
        Some(s) => handle::into_raw(AbstractShape::from_solid(s)),
        None => std::ptr::null_mut(),
    }
}

/// Returns true if `shape` wraps a vertex.
///
/// # Safety
/// `shape` must be NULL or a valid handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_is_vertex(shape: *const AbstractShape) -> bool {
    matches!(unsafe { handle::from_ref(shape) }.map(|s| &s.0), Some(SubShape::Vertex(_)))
}

/// Returns true if `shape` wraps an edge.
///
/// # Safety
/// `shape` must be NULL or a valid handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_is_edge(shape: *const AbstractShape) -> bool {
    matches!(unsafe { handle::from_ref(shape) }.map(|s| &s.0), Some(SubShape::Edge(_)))
}

/// Returns true if `shape` wraps a face.
///
/// # Safety
/// `shape` must be NULL or a valid handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_is_face(shape: *const AbstractShape) -> bool {
    matches!(unsafe { handle::from_ref(shape) }.map(|s| &s.0), Some(SubShape::Face(_)))
}

/// Returns true if `shape` wraps a shell.
///
/// # Safety
/// `shape` must be NULL or a valid handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_is_shell(shape: *const AbstractShape) -> bool {
    matches!(unsafe { handle::from_ref(shape) }.map(|s| &s.0), Some(SubShape::Shell(_)))
}

/// Returns true if `shape` wraps a solid.
///
/// # Safety
/// `shape` must be NULL or a valid handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_is_solid(shape: *const AbstractShape) -> bool {
    matches!(unsafe { handle::from_ref(shape) }.map(|s| &s.0), Some(SubShape::Solid(_)))
}

/// Consume `shape` and return the wrapped vertex, or NULL if it is not a vertex
/// (the `AbstractShape` is consumed regardless).
///
/// # Safety
/// `shape` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_into_vertex(
    shape: *mut AbstractShape,
) -> *mut TruckVertex {
    match unsafe { handle::take_raw(shape) }.map(|s| s.0) {
        Some(SubShape::Vertex(v)) => handle::into_raw(v),
        _ => std::ptr::null_mut(),
    }
}

/// Consume `shape` and return the wrapped edge, or NULL if not an edge.
///
/// # Safety
/// `shape` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_into_edge(
    shape: *mut AbstractShape,
) -> *mut TruckEdge {
    match unsafe { handle::take_raw(shape) }.map(|s| s.0) {
        Some(SubShape::Edge(e)) => handle::into_raw(e),
        _ => std::ptr::null_mut(),
    }
}

/// Consume `shape` and return the wrapped face, or NULL if not a face.
///
/// # Safety
/// `shape` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_into_face(
    shape: *mut AbstractShape,
) -> *mut TruckFace {
    match unsafe { handle::take_raw(shape) }.map(|s| s.0) {
        Some(SubShape::Face(f)) => handle::into_raw(f),
        _ => std::ptr::null_mut(),
    }
}

/// Consume `shape` and return the wrapped shell, or NULL if not a shell.
///
/// # Safety
/// `shape` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_into_shell(
    shape: *mut AbstractShape,
) -> *mut TruckShell {
    match unsafe { handle::take_raw(shape) }.map(|s| s.0) {
        Some(SubShape::Shell(s)) => handle::into_raw(s),
        _ => std::ptr::null_mut(),
    }
}

/// Consume `shape` and return the wrapped solid, or NULL if not a solid.
///
/// # Safety
/// `shape` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_into_solid(
    shape: *mut AbstractShape,
) -> *mut TruckSolid {
    match unsafe { handle::take_raw(shape) }.map(|s| s.0) {
        Some(SubShape::Solid(s)) => handle::into_raw(s),
        _ => std::ptr::null_mut(),
    }
}

/// Free an `AbstractShape` handle. Idempotent.
///
/// # Safety
/// `shape` must be NULL or a valid, owned handle.
#[no_mangle]
pub unsafe extern "C" fn truck_abstractshape_free(shape: *mut AbstractShape) {
    match unsafe { handle::take_raw(shape) } {
        Some(s) => drop(s),
        None => {}
    }
}

// ---------------------------------------------------------------------------
// sweep + transform (operate on AbstractShape)
// ---------------------------------------------------------------------------

/// Sweep `shape` by translation vector `vec` (`[x, y, z]`).
///
/// Result type depends on input (truck 0.6.0 `Sweep` mapping):
/// vertex→edge, edge→face, face→solid. Sweeping a shell or solid is an error
/// (shell sweep yields multiple solids; solid cannot be swept).
///
/// On success writes a new `AbstractShape` to `*out`; on failure writes an
/// error handle to `*err`.
///
/// # Safety
/// `shape` must be NULL or valid; `vec` must be NULL or valid for `vec_len`
/// f64s; `out`/`err` valid or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_tsweep(
    shape: *const AbstractShape,
    vec: *const f64,
    vec_len: usize,
    out: *mut *mut AbstractShape,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: caller guarantees shape is NULL or valid.
    let s = match unsafe { handle::from_ref(shape) } {
        Some(s) => s,
        None => return false,
    };
    let v = match unsafe { read_vec3(vec, vec_len) } {
        Some(v) => Vector3::new(v[0], v[1], v[2]),
        None => {
            if !err.is_null() {
                // SAFETY: err points to writable storage.
                unsafe { *err = handle::into_raw(TruckError::new("vec must be 3 f64s")) };
            }
            return false;
        }
    };
    let res = crate::error::truck_guard!(|| -> Result<AbstractShape, TruckError> {
        Ok(match &s.0 {
            SubShape::Vertex(vx) => AbstractShape::from_edge(TruckEdge(builder::tsweep(&vx.0, v))),
            SubShape::Edge(e) => AbstractShape::from_face(TruckFace(builder::tsweep(&e.0, v))),
            SubShape::Face(f) => AbstractShape::from_solid(TruckSolid(builder::tsweep(&f.0, v))),
            SubShape::Shell(_) => {
                return Err(TruckError::new("cannot tsweep a Shell (multi-result)"));
            }
            SubShape::Solid(_) => {
                return Err(TruckError::new("cannot tsweep a Solid"));
            }
        })
    });
    crate::truck_deliver!(res, err, |a: AbstractShape| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = handle::into_raw(a) };
    })
}

/// Sweep `shape` by rotation about `axis` through `origin` by `angle` radians.
///
/// `axis` must be normalized. truck 0.6.0 `rsweep` (via `ClosedSweep`) maps:
/// edge→shell, face→solid. Rotating a vertex/shell/solid is unsupported (the
/// result would be a Wire/multi-solid, which this ABI does not expose).
///
/// # Safety
/// `shape`, `origin`, `axis` NULL-or-valid; `out`/`err` valid or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_rsweep(
    shape: *const AbstractShape,
    origin: *const f64,
    origin_len: usize,
    axis: *const f64,
    axis_len: usize,
    angle: f64,
    out: *mut *mut AbstractShape,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    let s = match unsafe { handle::from_ref(shape) } {
        Some(s) => s,
        None => return false,
    };
    let o = match unsafe { read_vec3(origin, origin_len) } {
        Some(o) => Point3::new(o[0], o[1], o[2]),
        None => return false,
    };
    let a = match unsafe { read_vec3(axis, axis_len) } {
        Some(a) => Vector3::new(a[0], a[1], a[2]),
        None => return false,
    };
    let res = crate::error::truck_guard!(|| -> Result<AbstractShape, TruckError> {
        Ok(match &s.0 {
            SubShape::Edge(e) => AbstractShape::from_shell(TruckShell(builder::rsweep(
                &e.0, o, a, truck_modeling::Rad(angle),
            ))),
            SubShape::Face(f) => AbstractShape::from_solid(TruckSolid(builder::rsweep(
                &f.0, o, a, truck_modeling::Rad(angle),
            ))),
            SubShape::Vertex(_) => {
                return Err(TruckError::new("rsweep of a Vertex yields a Wire, not exposed"));
            }
            SubShape::Shell(_) | SubShape::Solid(_) => {
                return Err(TruckError::new("cannot rsweep a Shell or Solid"));
            }
        })
    });
    crate::truck_deliver!(res, err, |a: AbstractShape| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = handle::into_raw(a) };
    })
}

/// Translate `shape` by `vec` (`[x, y, z]`); returns the same shape type.
///
/// # Safety
/// `shape`/`vec` NULL-or-valid; `out`/`err` valid or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_translated(
    shape: *const AbstractShape,
    vec: *const f64,
    vec_len: usize,
    out: *mut *mut AbstractShape,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    let s = match unsafe { handle::from_ref(shape) } {
        Some(s) => s,
        None => return false,
    };
    let v = match unsafe { read_vec3(vec, vec_len) } {
        Some(v) => Vector3::new(v[0], v[1], v[2]),
        None => return false,
    };
    let res = crate::error::truck_guard!(|| -> Result<AbstractShape, TruckError> {
        Ok(match &s.0 {
            SubShape::Vertex(vx) => AbstractShape::from_vertex(TruckVertex(builder::translated(&vx.0, v))),
            SubShape::Edge(e) => AbstractShape::from_edge(TruckEdge(builder::translated(&e.0, v))),
            SubShape::Face(f) => AbstractShape::from_face(TruckFace(builder::translated(&f.0, v))),
            SubShape::Shell(sh) => AbstractShape::from_shell(TruckShell(builder::translated(&sh.0, v))),
            SubShape::Solid(so) => AbstractShape::from_solid(TruckSolid(builder::translated(&so.0, v))),
        })
    });
    crate::truck_deliver!(res, err, |a: AbstractShape| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = handle::into_raw(a) };
    })
}

/// Rotate `shape` about `axis` through `origin` by `angle` radians; same type.
/// `axis` must be normalized.
///
/// # Safety
/// All pointer args NULL-or-valid per their roles.
#[no_mangle]
pub unsafe extern "C" fn truck_rotated(
    shape: *const AbstractShape,
    origin: *const f64,
    origin_len: usize,
    axis: *const f64,
    axis_len: usize,
    angle: f64,
    out: *mut *mut AbstractShape,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    let s = match unsafe { handle::from_ref(shape) } {
        Some(s) => s,
        None => return false,
    };
    let o = match unsafe { read_vec3(origin, origin_len) } {
        Some(o) => Point3::new(o[0], o[1], o[2]),
        None => return false,
    };
    let a = match unsafe { read_vec3(axis, axis_len) } {
        Some(a) => Vector3::new(a[0], a[1], a[2]),
        None => return false,
    };
    let res = crate::error::truck_guard!(|| -> Result<AbstractShape, TruckError> {
        Ok(match &s.0 {
            SubShape::Vertex(vx) => AbstractShape::from_vertex(TruckVertex(builder::rotated(&vx.0, o, a, truck_modeling::Rad(angle)))),
            SubShape::Edge(e) => AbstractShape::from_edge(TruckEdge(builder::rotated(&e.0, o, a, truck_modeling::Rad(angle)))),
            SubShape::Face(f) => AbstractShape::from_face(TruckFace(builder::rotated(&f.0, o, a, truck_modeling::Rad(angle)))),
            SubShape::Shell(sh) => AbstractShape::from_shell(TruckShell(builder::rotated(&sh.0, o, a, truck_modeling::Rad(angle)))),
            SubShape::Solid(so) => AbstractShape::from_solid(TruckSolid(builder::rotated(&so.0, o, a, truck_modeling::Rad(angle)))),
        })
    });
    crate::truck_deliver!(res, err, |a: AbstractShape| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = handle::into_raw(a) };
    })
}

/// Scale `shape` about `origin` by `scalars` (`[sx, sy, sz]`); same type.
///
/// # Safety
/// All pointer args NULL-or-valid per their roles.
#[no_mangle]
pub unsafe extern "C" fn truck_scaled(
    shape: *const AbstractShape,
    origin: *const f64,
    origin_len: usize,
    scalars: *const f64,
    scalars_len: usize,
    out: *mut *mut AbstractShape,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    let s = match unsafe { handle::from_ref(shape) } {
        Some(s) => s,
        None => return false,
    };
    let o = match unsafe { read_vec3(origin, origin_len) } {
        Some(o) => Point3::new(o[0], o[1], o[2]),
        None => return false,
    };
    let sc = match unsafe { read_vec3(scalars, scalars_len) } {
        Some(sc) => Vector3::new(sc[0], sc[1], sc[2]),
        None => return false,
    };
    let res = crate::error::truck_guard!(|| -> Result<AbstractShape, TruckError> {
        Ok(match &s.0 {
            SubShape::Vertex(vx) => AbstractShape::from_vertex(TruckVertex(builder::scaled(&vx.0, o, sc))),
            SubShape::Edge(e) => AbstractShape::from_edge(TruckEdge(builder::scaled(&e.0, o, sc))),
            SubShape::Face(f) => AbstractShape::from_face(TruckFace(builder::scaled(&f.0, o, sc))),
            SubShape::Shell(sh) => AbstractShape::from_shell(TruckShell(builder::scaled(&sh.0, o, sc))),
            SubShape::Solid(so) => AbstractShape::from_solid(TruckSolid(builder::scaled(&so.0, o, sc))),
        })
    });
    crate::truck_deliver!(res, err, |a: AbstractShape| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = handle::into_raw(a) };
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    // Bring helpers into scope for the 4d tessellation/error tests.
    use crate::error::truck_error_free;
    use crate::polymesh::truck_polygonmesh_free;

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

    // ---- stage 4d-0: meshalgo compatibility smoke test ---------------------
    // Go/no-go gate: verifies that truck_modeling 0.6.0's Curve/Surface satisfy
    // truck_meshalgo 0.4.0's PolylineableCurve/MeshableSurface bounds, by
    // driving the full builder chain vertex -> edge -> face -> solid and then
    // tessellating the solid into a PolygonMesh. If this compiles AND runs, the
    // version combination is compatible and stage 4d-1 can proceed.
    #[test]
    fn meshalgo_compatibility_smoke() {
        use truck_meshalgo::tessellation::{MeshedShape, RobustMeshableShape};

        let v: truck_modeling::Vertex = truck_modeling::builder::vertex(Point3::new(0.0, 0.0, 0.0));
        let e: truck_modeling::Edge =
            truck_modeling::builder::tsweep(&v, truck_modeling::Vector3::new(1.0, 0.0, 0.0));
        let f: truck_modeling::Face =
            truck_modeling::builder::tsweep(&e, truck_modeling::Vector3::new(0.0, 1.0, 0.0));
        let s: truck_modeling::Solid =
            truck_modeling::builder::tsweep(&f, truck_modeling::Vector3::new(0.0, 0.0, 1.0));

        // Tessellate the solid's first boundary shell into a polygon mesh.
        let meshed = s.robust_triangulation(0.01);
        let shell = &meshed.boundaries()[0];
        let polygon: truck_polymesh::PolygonMesh = shell.to_polygon();
        assert!(!polygon.positions().is_empty(), "tessellated mesh should have positions");
    }

    // ---- stage 4d-1: Shell / Solid / AbstractShape / sweep -----------------

    #[test]
    fn tsweep_chain_vertex_to_solid() {
        // vertex -> edge -> face -> solid, each via AbstractShape.
        let v = truck_vertex_new(0.0, 0.0, 0.0);
        let vec_x = [1.0, 0.0, 0.0];
        // SAFETY: v valid.
        let s0 = unsafe { truck_vertex_upcast(v) };
        assert!(!s0.is_null());

        let mut s1 = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        // SAFETY: s0 valid; vec_x valid.
        let ok = unsafe { truck_tsweep(s0, vec_x.as_ptr(), 3, &mut s1, &mut err) };
        assert!(ok, "vertex tsweep should succeed");
        // SAFETY: s1 valid.
        assert!(unsafe { truck_abstractshape_is_edge(s1) });

        let mut s2 = std::ptr::null_mut();
        let vec_y = [0.0, 1.0, 0.0];
        // SAFETY: s1 valid; vec_y valid.
        assert!(unsafe { truck_tsweep(s1, vec_y.as_ptr(), 3, &mut s2, &mut err) });
        // SAFETY: s2 valid.
        assert!(unsafe { truck_abstractshape_is_face(s2) });

        let mut s3 = std::ptr::null_mut();
        let vec_z = [0.0, 0.0, 1.0];
        // SAFETY: s2 valid; vec_z valid.
        assert!(unsafe { truck_tsweep(s2, vec_z.as_ptr(), 3, &mut s3, &mut err) });
        // SAFETY: s3 valid.
        assert!(unsafe { truck_abstractshape_is_solid(s3) });

        unsafe {
            truck_abstractshape_free(s3);
            truck_abstractshape_free(s2);
            truck_abstractshape_free(s1);
            truck_abstractshape_free(s0);
        }
    }

    #[test]
    fn tsweep_solid_is_error() {
        let v = truck_vertex_new(0.0, 0.0, 0.0);
        // SAFETY: v valid.
        let s0 = unsafe { truck_vertex_upcast(v) };
        let mut s1 = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        let vec = [1.0, 0.0, 0.0];
        // SAFETY: s0 valid.
        unsafe { truck_tsweep(s0, vec.as_ptr(), 3, &mut s1, &mut err) }; // vertex->edge
        let mut s2 = std::ptr::null_mut();
        unsafe { truck_tsweep(s1, [0.0, 1.0, 0.0].as_ptr(), 3, &mut s2, &mut err) }; // edge->face
        let mut s3 = std::ptr::null_mut();
        unsafe { truck_tsweep(s2, [0.0, 0.0, 1.0].as_ptr(), 3, &mut s3, &mut err) }; // face->solid

        // tsweep a solid -> error
        let mut s4 = std::ptr::null_mut();
        err = std::ptr::null_mut();
        // SAFETY: s3 valid (a solid).
        let ok = unsafe { truck_tsweep(s3, [1.0, 0.0, 0.0].as_ptr(), 3, &mut s4, &mut err) };
        assert!(!ok, "tsweep of a solid should fail");
        assert!(!err.is_null());
        unsafe {
            truck_error_free(err);
            truck_abstractshape_free(s4);
            truck_abstractshape_free(s3);
            truck_abstractshape_free(s2);
            truck_abstractshape_free(s1);
            truck_abstractshape_free(s0);
        }
    }

    #[test]
    fn solid_to_polygon_via_tsweep() {
        // vertex -> edge -> face -> solid, then tessellate the solid.
        let v = truck_vertex_new(0.0, 0.0, 0.0);
        // SAFETY: v valid.
        let s0 = unsafe { truck_vertex_upcast(v) };
        let mut s1 = std::ptr::null_mut();
        let mut s2 = std::ptr::null_mut();
        let mut s3 = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        // SAFETY: chain of valid shapes; tsweep borrows input, so s0/s1/s2 stay alive.
        unsafe {
            truck_tsweep(s0, [1.0, 0.0, 0.0].as_ptr(), 3, &mut s1, &mut err);
            truck_tsweep(s1, [0.0, 1.0, 0.0].as_ptr(), 3, &mut s2, &mut err);
            truck_tsweep(s2, [0.0, 0.0, 1.0].as_ptr(), 3, &mut s3, &mut err);
        }
        // s3 wraps a solid; extract it (consumes s3).
        // SAFETY: s3 valid (a solid).
        let solid = unsafe { truck_abstractshape_into_solid(s3) };
        assert!(!solid.is_null());

        let mut mesh: *mut TruckPolygonMesh = std::ptr::null_mut();
        let mut err2 = std::ptr::null_mut();
        // SAFETY: solid valid.
        let ok = unsafe { truck_solid_to_polygon(solid, 0.01, &mut mesh, &mut err2) };
        assert!(ok, "solid_to_polygon should succeed");
        assert!(!mesh.is_null());
        unsafe {
            truck_polygonmesh_free(mesh);
            truck_solid_free(solid);
            truck_abstractshape_free(s2);
            truck_abstractshape_free(s1);
            truck_abstractshape_free(s0);
        }
    }

    #[test]
    fn shell_from_faces_and_to_polygon() {
        // Build a face via homotopy, wrap in a shell, tessellate the shell.
        let v0 = truck_vertex_new(0.0, 0.0, 0.0);
        let v1 = truck_vertex_new(1.0, 0.0, 0.0);
        let v2 = truck_vertex_new(0.0, 0.0, 1.0);
        let v3 = truck_vertex_new(1.0, 0.0, 1.0);
        // SAFETY: vertices valid.
        let e0 = unsafe { truck_edge_line(v0, v1) };
        let e1 = unsafe { truck_edge_line(v2, v3) };
        // SAFETY: edges valid.
        let face = unsafe { truck_face_homotopy(e0, e1) };
        let faces = [face];
        // SAFETY: faces array valid; face consumed.
        let shell = unsafe { truck_shell_from_faces(faces.as_ptr(), 1) };
        assert!(!shell.is_null());

        let mut mesh: *mut TruckPolygonMesh = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        // SAFETY: shell valid.
        let ok = unsafe { truck_shell_to_polygon(shell, 0.01, &mut mesh, &mut err) };
        assert!(ok, "shell_to_polygon should succeed");
        assert!(!mesh.is_null());
        unsafe {
            truck_polygonmesh_free(mesh);
            truck_shell_free(shell);
            truck_edge_free(e0);
            truck_edge_free(e1);
            truck_vertex_free(v0);
            truck_vertex_free(v1);
            truck_vertex_free(v2);
            truck_vertex_free(v3);
        }
    }

    #[test]
    fn translated_preserves_type() {
        let v = truck_vertex_new(0.0, 0.0, 0.0);
        // SAFETY: v valid.
        let s = unsafe { truck_vertex_upcast(v) };
        let mut out = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        let vec = [5.0, 0.0, 0.0];
        // SAFETY: s valid; vec valid.
        let ok = unsafe { truck_translated(s, vec.as_ptr(), 3, &mut out, &mut err) };
        assert!(ok);
        // SAFETY: out valid.
        assert!(unsafe { truck_abstractshape_is_vertex(out) });
        // extract vertex and check its point moved
        // SAFETY: out valid (a vertex).
        let mv = unsafe { truck_abstractshape_into_vertex(out) };
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        // SAFETY: mv valid.
        unsafe { truck_vertex_point(mv, &mut arr) };
        // SAFETY: arr valid for len.
        let pts = unsafe { std::slice::from_raw_parts(arr.ptr, arr.len) };
        assert_eq!(pts, &[5.0, 0.0, 0.0]);
        unsafe {
            crate::handle::truck_f64array_free(arr);
            truck_vertex_free(mv);
            truck_abstractshape_free(s);
        }
    }

    #[test]
    fn abstractshape_into_wrong_type_yields_null() {
        let v = truck_vertex_new(0.0, 0.0, 0.0);
        // SAFETY: v valid.
        let s = unsafe { truck_vertex_upcast(v) };
        // into_edge on a vertex-shape -> NULL, shape still consumed
        // SAFETY: s valid.
        let edge = unsafe { truck_abstractshape_into_edge(s) };
        assert!(edge.is_null());
        // SAFETY: NULL is fine to free.
        unsafe { truck_edge_free(edge) };
    }

    #[test]
    fn shell_solid_free_null_safe() {
        // SAFETY: NULL is explicitly allowed.
        unsafe {
            truck_shell_free(std::ptr::null_mut());
            truck_solid_free(std::ptr::null_mut());
            truck_abstractshape_free(std::ptr::null_mut());
            truck_vertex_upcast(std::ptr::null_mut());
            truck_edge_upcast(std::ptr::null_mut());
        }
    }
}

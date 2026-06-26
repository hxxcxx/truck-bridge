//! `PolygonMesh` C ABI surface.
//!
//! Stage 2 gave the read-only minimum (`new_empty`, `bounding_box`, `free`).
//! Stage 3 (this file) adds the full IO surface:
//!   - `from_obj` / `to_obj` — Wavefront OBJ parsing & serialization
//!   - `from_stl` / `to_stl` — STL parsing & serialization (ascii/binary)
//!   - `to_buffer` — flattened, separated arrays for GPU upload
//!   - `merge` — combine two meshes
//!
//! ## Error convention
//!
//! Every failable function takes an `err: *mut *mut TruckError` out-parameter
//! and returns `bool`. On success the result is written into its out-parameter
//! and `true` is returned. On failure an error handle is written to `*err`
//! (only if `err` itself is non-NULL) and `false` is returned. All bodies run
//! under [`truck_guard!`] so a panic becomes an error instead of unwinding.
//!
//! ## `let ... else` is forbidden
//!
//! cbindgen 0.x's parser does not support `let ... else`. Use `match`. This
//! rule applies to the whole crate.

use crate::error::{self, TruckError};
use crate::handle::{
    self, TruckF32Array, TruckF64Array, TruckU32Array, TruckU8Array,
};

// truck's concrete mesh type, re-exported by lib.rs as the single monomorphized
// `PolygonMesh<StandardVertex, StandardAttributes>` form we use everywhere.

/// Opaque handle to a truck `PolygonMesh` (monomorphized to its default
/// `<StandardVertex, StandardAttributes>` form — the only form truck itself
/// uses in practice). C sees only `typedef struct TruckPolygonMesh ...;`.
#[derive(Debug)]
pub struct TruckPolygonMesh(pub(crate) crate::PolygonMesh);

/// STL format selector. Values must stay matched 1:1 with
/// [`truck_polymesh::stl::StlType`] via the conversion below; the explicit
/// `From` impl is the single source of truth (do not rely on discriminant
/// ordering).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruckStlType {
    /// Auto-detect on read (header bytes); always binary on write.
    Automatic = 0,
    /// ASCII STL.
    Ascii = 1,
    /// Binary STL.
    Binary = 2,
}

impl From<TruckStlType> for truck_polymesh::stl::StlType {
    fn from(t: TruckStlType) -> Self {
        use truck_polymesh::stl::StlType;
        match t {
            TruckStlType::Automatic => StlType::Automatic,
            TruckStlType::Ascii => StlType::Ascii,
            TruckStlType::Binary => StlType::Binary,
        }
    }
}

/// Flattened, **separated** per-vertex data for GPU upload.
///
/// Deliberately diverges from `truck-js` (which interleaves pos+uv+normal into
/// one `f32` stream): truck-bridge hands back four independent arrays so the
/// consumer can pack them however its renderer wants.
///
/// Invariants when produced by [`truck_polygonmesh_to_buffer`]:
///   - `positions.len == 3 * vertex_count` (xyz per vertex)
///   - `uv.len == 2 * vertex_count`
///   - `normal.len == 3 * vertex_count`
///   - `indices.len == 3 * triangle_count`
///
/// `vertex_count == 0` for an empty mesh; all four arrays are then empty.
/// Free the whole struct with [`truck_polygonbuffer_free`].
#[repr(C)]
#[derive(Debug)]
pub struct TruckPolygonBuffer {
    /// Vertex positions, xyz packed: `[x0,y0,z0, x1,y1,z1, ...]`.
    pub positions: TruckF64Array,
    /// Texture coordinates, uv packed: `[u0,v0, u1,v1, ...]`.
    pub uv: TruckF32Array,
    /// Vertex normals, xyz packed.
    pub normal: TruckF32Array,
    /// Triangle index list, 3 per triangle.
    pub indices: TruckU32Array,
}

// ---------------------------------------------------------------------------
// result-delivery helper
// ---------------------------------------------------------------------------

/// Drive a [`truck_guard!`] result into `(bool, *err)`:
///   - `Ok(v)`  → run `$ok(v)` (which writes into the success out-parameter).
///   - `Err(e)` → write the error handle to `*err` (if non-NULL), ignore v.
///
/// `$res` is a `Result<T, *mut TruckError>` as produced by `truck_guard!`.
macro_rules! truck_deliver {
    ($res:expr, $err:expr, $ok:expr) => {
        match $res {
            ::std::result::Result::Ok(v) => { $ok(v); true }
            ::std::result::Result::Err(e) => {
                if !$err.is_null() {
                    // SAFETY: caller guarantees err points to writable storage.
                    unsafe { *$err = e };
                } else {
                    // Nobody to receive it; free to avoid a leak.
                    // SAFETY: e is a freshly-allocated owning handle.
                    unsafe { crate::handle::take_raw::<TruckError>(e) };
                }
                false
            }
        }
    };
}
// (macro_rules! truck_deliver is in scope within this module without an import)

/// Wrap a fallible truck operation in an error-converting closure for
/// `truck_guard!`. Maps `Result<_, E: Display>` to `Result<_, TruckError>`.
fn lift<E: std::fmt::Display, T>(r: Result<T, E>) -> Result<T, TruckError> {
    r.map_err(|e| TruckError::new(format!("{e}")))
}

// ---------------------------------------------------------------------------
// Stage 2 API (preserved)
// ---------------------------------------------------------------------------

/// Create an empty polygon mesh (no positions, no faces).
///
/// Returns an owning handle; release it with [`truck_polygonmesh_free`].
#[no_mangle]
pub extern "C" fn truck_polygonmesh_new_empty() -> *mut TruckPolygonMesh {
    handle::into_raw(TruckPolygonMesh(crate::PolygonMesh::default()))
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

// ---------------------------------------------------------------------------
// Stage 3 API — IO
// ---------------------------------------------------------------------------

/// Parse Wavefront OBJ bytes into a new mesh.
///
/// On success writes a new handle to `*out_mesh` (caller frees with
/// [`truck_polygonmesh_free`]); on failure writes an error handle to `*err`
/// (if `err` is non-NULL; free with [`truck_error_free`]) and returns `false`.
///
/// # Safety
/// `data` must be valid for `len` bytes (NULL only allowed if `len == 0`).
/// `out_mesh` and `err` must be valid writable pointers or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_from_obj(
    data: *const u8,
    len: usize,
    out_mesh: *mut *mut TruckPolygonMesh,
    err: *mut *mut TruckError,
) -> bool {
    if out_mesh.is_null() {
        return false;
    }
    let bytes: &[u8] = if data.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: caller guarantees data is valid for len bytes.
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let res = error::truck_guard!(|| lift(truck_polymesh::obj::read::<&[u8]>(bytes)));
    truck_deliver!(res, err, |m: crate::PolygonMesh| {
        // SAFETY: out_mesh checked non-NULL above.
        unsafe { *out_mesh = handle::into_raw(TruckPolygonMesh(m)) };
    })
}

/// Parse STL bytes into a new mesh. See [`truck_polygonmesh_from_obj`] for the
/// out-parameter / error contract.
///
/// # Safety
/// Same as [`truck_polygonmesh_from_obj`].
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_from_stl(
    data: *const u8,
    len: usize,
    stl_type: TruckStlType,
    out_mesh: *mut *mut TruckPolygonMesh,
    err: *mut *mut TruckError,
) -> bool {
    if out_mesh.is_null() {
        return false;
    }
    let bytes: &[u8] = if data.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: caller guarantees data is valid for len bytes.
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    let res = error::truck_guard!(|| {
        lift(truck_polymesh::stl::read::<&[u8]>(bytes, stl_type.into()))
    });
    truck_deliver!(res, err, |m: crate::PolygonMesh| {
        // SAFETY: out_mesh checked non-NULL above.
        unsafe { *out_mesh = handle::into_raw(TruckPolygonMesh(m)) };
    })
}

/// Serialize a mesh to Wavefront OBJ bytes.
///
/// On success writes the bytes to `*out` (caller frees with
/// [`truck_u8array_free`]); on failure writes an error handle to `*err`.
///
/// # Safety
/// `mesh` must be NULL or a valid handle. `out` and `err` must be valid
/// writable pointers or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_to_obj(
    mesh: *const TruckPolygonMesh,
    out: *mut TruckU8Array,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: caller guarantees mesh is NULL or a valid handle.
    let m = match unsafe { handle::from_ref(mesh) } {
        Some(m) => m,
        None => return false,
    };
    let res = error::truck_guard!(|| {
        let mut buf = Vec::new();
        lift(truck_polymesh::obj::write(&m.0, &mut buf))?;
        Ok::<_, TruckError>(buf)
    });
    truck_deliver!(res, err, |buf: Vec<u8>| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = TruckU8Array::from(buf) };
    })
}

/// Serialize a mesh to STL bytes. See [`truck_polygonmesh_to_obj`] for the
/// out-parameter / error contract. `StlType::Automatic` writes binary.
///
/// # Safety
/// Same as [`truck_polygonmesh_to_obj`].
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_to_stl(
    mesh: *const TruckPolygonMesh,
    stl_type: TruckStlType,
    out: *mut TruckU8Array,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: caller guarantees mesh is NULL or a valid handle.
    let m = match unsafe { handle::from_ref(mesh) } {
        Some(m) => m,
        None => return false,
    };
    let res = error::truck_guard!(|| {
        let mut buf = Vec::new();
        lift(truck_polymesh::stl::write(&m.0, &mut buf, stl_type.into()))?;
        Ok::<_, TruckError>(buf)
    });
    truck_deliver!(res, err, |buf: Vec<u8>| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = TruckU8Array::from(buf) };
    })
}

// ---------------------------------------------------------------------------
// Stage 3 API — buffer + merge
// ---------------------------------------------------------------------------

/// Flatten the mesh into separated per-vertex arrays for GPU upload.
///
/// Produces a [`TruckPolygonBuffer`] with independent `positions` (f64),
/// `uv` (f32), `normal` (f32) and `indices` (u32). The mesh is fully expanded:
/// each triangle vertex is emitted as its own attribute entry, so the three
/// attribute arrays share one index space and all have `vertex_count` logical
/// vertices (`positions.len == 3 * vertex_count`, etc.).
///
/// Free the result with [`truck_polygonbuffer_free`].
///
/// # Safety
/// `mesh` must be NULL or a valid handle. `out` and `err` must be valid
/// writable pointers or NULL.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_to_buffer(
    mesh: *const TruckPolygonMesh,
    out: *mut TruckPolygonBuffer,
    err: *mut *mut TruckError,
) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: caller guarantees mesh is NULL or a valid handle.
    let m = match unsafe { handle::from_ref(mesh) } {
        Some(m) => m,
        None => return false,
    };
    let res = error::truck_guard!(|| {
        // Expand each face-vertex into its own (pos, uv, normal) tuple.
        let exp = m.0.expands(|attr| {
            let p = attr.position;
            let uv = attr.uv_coord.unwrap_or_else(|| {
                truck_polymesh::base::Vector2::new(0.0, 0.0)
            });
            let n = attr.normal.unwrap_or_else(|| {
                truck_polymesh::base::Vector3::new(0.0, 0.0, 0.0)
            });
            (p, uv, n)
        });
        let n_vert = exp.attributes().len();
        let mut positions = Vec::with_capacity(n_vert * 3);
        let mut uv = Vec::with_capacity(n_vert * 2);
        let mut normal = Vec::with_capacity(n_vert * 3);
        for (p, t, nm) in exp.attributes().iter() {
            positions.push(p[0] as f64);
            positions.push(p[1] as f64);
            positions.push(p[2] as f64);
            uv.push(t[0] as f32);
            uv.push(t[1] as f32);
            normal.push(nm[0] as f32);
            normal.push(nm[1] as f32);
            normal.push(nm[2] as f32);
        }
        let indices: Vec<u32> = exp.faces().triangle_iter().flatten().map(|i| i as u32).collect();
        Ok::<_, TruckError>(TruckPolygonBuffer {
            positions: TruckF64Array::from(positions),
            uv: TruckF32Array::from(uv),
            normal: TruckF32Array::from(normal),
            indices: TruckU32Array::from(indices),
        })
    });
    truck_deliver!(res, err, |buf: TruckPolygonBuffer| {
        // SAFETY: out checked non-NULL above.
        unsafe { *out = buf };
    })
}

/// Merge `src` into `dst` in place.
///
/// **Ownership:** `src` is *consumed* — its handle is freed by this call. Do
/// not use or free `src` afterwards (doing so is a benign no-op on a NULL
/// handle, since `take_raw`/`*_free` tolerate NULL, but the mesh it held is
/// now part of `dst`).
///
/// Returns `true` on success, `false` if either handle is NULL.
///
/// # Safety
/// `dst` and `src` must each be NULL or valid handles. `src` must not be
/// accessed after this call.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonmesh_merge(
    dst: *mut TruckPolygonMesh,
    src: *mut TruckPolygonMesh,
    err: *mut *mut TruckError,
) -> bool {
    // SAFETY: caller guarantees dst is NULL or a valid handle.
    let dst_ref = match unsafe { handle::from_mut(dst) } {
        Some(d) => d,
        None => return false,
    };
    // Consume src: take ownership of its inner PolygonMesh (drops the handle).
    // SAFETY: caller guarantees src is NULL or a valid, owned handle.
    let src_mesh = match unsafe { handle::take_raw(src) } {
        Some(s) => s.0,
        None => return false,
    };
    let res = error::truck_guard!(|| {
        dst_ref.0.merge(src_mesh);
        Ok::<_, TruckError>(())
    });
    truck_deliver!(res, err, |_v: ()| {})
}

/// Free a [`TruckPolygonBuffer`] and its four internal arrays. Idempotent: each
/// array tolerates being empty.
///
/// # Safety
/// `buf` must describe allocations previously produced by truck-bridge (each
/// array may be the empty value), and must not already have been freed.
#[no_mangle]
pub unsafe extern "C" fn truck_polygonbuffer_free(buf: TruckPolygonBuffer) {
    // SAFETY: each array came from a Truck*Array::from (or empty), cap == len.
    unsafe {
        crate::handle::truck_f32array_free(buf.normal);
        crate::handle::truck_f32array_free(buf.uv);
        crate::handle::truck_u32array_free(buf.indices);
        crate::handle::truck_f64array_free(buf.positions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal single-triangle OBJ.
    const TRI_OBJ: &[u8] = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1 2 3\n";

    fn from_obj(bytes: &[u8]) -> *mut TruckPolygonMesh {
        let mut out: *mut TruckPolygonMesh = std::ptr::null_mut();
        let mut err: *mut TruckError = std::ptr::null_mut();
        // SAFETY: bytes is a valid slice; out/err are valid pointers.
        let ok = unsafe {
            truck_polygonmesh_from_obj(bytes.as_ptr(), bytes.len(), &mut out, &mut err)
        };
        assert!(ok, "from_obj failed unexpectedly");
        assert!(out != std::ptr::null_mut());
        assert!(err.is_null());
        out
    }

    #[test]
    fn obj_roundtrip_preserves_positions() {
        let m1 = from_obj(TRI_OBJ);
        // serialize back
        let mut bytes = TruckU8Array { ptr: std::ptr::null_mut(), len: 0 };
        let mut err: *mut TruckError = std::ptr::null_mut();
        // SAFETY: m1 valid; bytes/err valid.
        let ok = unsafe { truck_polygonmesh_to_obj(m1, &mut bytes, &mut err) };
        assert!(ok);
        assert!(bytes.len > 0);
        // re-parse
        // SAFETY: bytes.ptr valid for bytes.len.
        let slice = unsafe { std::slice::from_raw_parts(bytes.ptr, bytes.len) };
        let m2 = from_obj(slice);
        // compare positions via bounding box (sufficient for this tiny mesh)
        let mut b1 = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        let mut b2 = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        unsafe {
            truck_polygonmesh_bounding_box(m1, &mut b1);
            truck_polygonmesh_bounding_box(m2, &mut b2);
        }
        // SAFETY: valid for b1.len/b2.len.
        let s1 = unsafe { std::slice::from_raw_parts(b1.ptr, b1.len) };
        let s2 = unsafe { std::slice::from_raw_parts(b2.ptr, b2.len) };
        assert_eq!(s1, s2);
        unsafe {
            crate::handle::truck_f64array_free(b1);
            crate::handle::truck_f64array_free(b2);
            crate::handle::truck_u8array_free(bytes);
            truck_polygonmesh_free(m1);
            truck_polygonmesh_free(m2);
        }
    }

    #[test]
    fn from_obj_garbage_yields_empty_mesh() {
        // OBJ parsing is permissive: unrecognized lines are skipped, so garbage
        // text parses to an empty mesh rather than an error. This documents
        // that behavior.
        let mut out: *mut TruckPolygonMesh = std::ptr::null_mut();
        let mut err: *mut TruckError = std::ptr::null_mut();
        let garbage = b"not an obj at all!!!";
        // SAFETY: garbage is a valid slice; out/err valid.
        let ok = unsafe {
            truck_polygonmesh_from_obj(garbage.as_ptr(), garbage.len(), &mut out, &mut err)
        };
        assert!(ok, "garbage OBJ should parse to an empty mesh, not fail");
        assert!(err.is_null());
        // The resulting mesh has no vertices -> bounding box is the inverted
        // infinity box.
        let mut bbox = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        unsafe {
            truck_polygonmesh_bounding_box(out, &mut bbox);
            crate::handle::truck_f64array_free(bbox);
            truck_polygonmesh_free(out);
        }
    }

    #[test]
    fn from_stl_truncated_yields_error() {
        // A truncated binary STL (too few bytes for the 84-byte header) is a
        // genuine parse error, exercising the err out-parameter path.
        let mut out: *mut TruckPolygonMesh = std::ptr::null_mut();
        let mut err: *mut TruckError = std::ptr::null_mut();
        let truncated = b"\x00\x00\x00\x00truncated";
        // SAFETY: truncated is a valid slice; out/err valid.
        let ok = unsafe {
            truck_polygonmesh_from_stl(
                truncated.as_ptr(),
                truncated.len(),
                TruckStlType::Binary,
                &mut out,
                &mut err,
            )
        };
        assert!(!ok, "truncated STL should fail to parse");
        assert!(out.is_null());
        assert!(!err.is_null(), "an error handle should be produced");
        // The error must carry a non-empty message.
        // SAFETY: err is valid & owning.
        let s = unsafe { crate::error::truck_error_message(err) };
        // SAFETY: s.ptr valid for s.len.
        let mbytes = unsafe { std::slice::from_raw_parts(s.ptr, s.len) };
        assert!(!mbytes.is_empty(), "error message should be non-empty");
        unsafe {
            crate::handle::truck_str_free(s);
            crate::error::truck_error_free(err);
        }
    }

    #[test]
    fn to_buffer_single_triangle() {
        let m = from_obj(TRI_OBJ);
        let mut buf = TruckPolygonBuffer {
            positions: TruckF64Array { ptr: std::ptr::null_mut(), len: 0 },
            uv: TruckF32Array { ptr: std::ptr::null_mut(), len: 0 },
            normal: TruckF32Array { ptr: std::ptr::null_mut(), len: 0 },
            indices: TruckU32Array { ptr: std::ptr::null_mut(), len: 0 },
        };
        let mut err: *mut TruckError = std::ptr::null_mut();
        // SAFETY: m valid; buf/err valid.
        let ok = unsafe { truck_polygonmesh_to_buffer(m, &mut buf, &mut err) };
        assert!(ok);
        assert_eq!(buf.positions.len, 9); // 3 verts * 3
        assert_eq!(buf.uv.len, 6); // 3 verts * 2
        assert_eq!(buf.normal.len, 9);
        assert_eq!(buf.indices.len, 3); // 1 triangle
        unsafe {
            truck_polygonbuffer_free(buf);
            truck_polygonmesh_free(m);
        }
    }

    #[test]
    fn merge_combines_two_meshes() {
        let a = from_obj(TRI_OBJ);
        let b = from_obj(TRI_OBJ);
        let mut err: *mut TruckError = std::ptr::null_mut();
        // SAFETY: a, b valid handles; err valid.
        let ok = unsafe { truck_polygonmesh_merge(a, b, &mut err) };
        assert!(ok);
        // buffer of merged mesh should have 2 triangles
        let mut buf = TruckPolygonBuffer {
            positions: TruckF64Array { ptr: std::ptr::null_mut(), len: 0 },
            uv: TruckF32Array { ptr: std::ptr::null_mut(), len: 0 },
            normal: TruckF32Array { ptr: std::ptr::null_mut(), len: 0 },
            indices: TruckU32Array { ptr: std::ptr::null_mut(), len: 0 },
        };
        let mut err2: *mut TruckError = std::ptr::null_mut();
        unsafe {
            truck_polygonmesh_to_buffer(a, &mut buf, &mut err2);
            assert_eq!(buf.indices.len, 6); // 2 triangles
            truck_polygonbuffer_free(buf);
            truck_polygonmesh_free(a); // b already consumed by merge
        }
    }

    #[test]
    fn stl_binary_roundtrip() {
        let m1 = from_obj(TRI_OBJ);
        // to binary stl
        let mut bytes = TruckU8Array { ptr: std::ptr::null_mut(), len: 0 };
        let mut err: *mut TruckError = std::ptr::null_mut();
        // SAFETY: m1 valid.
        let ok = unsafe {
            truck_polygonmesh_to_stl(m1, TruckStlType::Binary, &mut bytes, &mut err)
        };
        assert!(ok);
        assert!(bytes.len > 0);
        // from binary stl
        // SAFETY: bytes.ptr valid for bytes.len.
        let slice = unsafe { std::slice::from_raw_parts(bytes.ptr, bytes.len) };
        let mut m2_raw: *mut TruckPolygonMesh = std::ptr::null_mut();
        let mut err2: *mut TruckError = std::ptr::null_mut();
        // SAFETY: slice valid.
        let ok2 = unsafe {
            truck_polygonmesh_from_stl(
                slice.as_ptr(),
                slice.len(),
                TruckStlType::Binary,
                &mut m2_raw,
                &mut err2,
            )
        };
        assert!(ok2);
        assert!(!m2_raw.is_null());
        unsafe {
            crate::handle::truck_u8array_free(bytes);
            truck_polygonmesh_free(m1);
            truck_polygonmesh_free(m2_raw);
        }
    }

    // ---- stage 2 regression tests (kept) ----

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
        assert!(s[0].is_infinite() && s[0].is_sign_positive());
        assert!(s[3].is_infinite() && s[3].is_sign_negative());
        unsafe {
            crate::handle::truck_f64array_free(arr);
            truck_polygonmesh_free(m);
        }
    }

    #[test]
    fn null_arguments_return_false() {
        let m = truck_polygonmesh_new_empty();
        let mut arr = TruckF64Array { ptr: std::ptr::null_mut(), len: 0 };
        assert!(!unsafe { truck_polygonmesh_bounding_box(std::ptr::null(), &mut arr) });
        assert!(!unsafe { truck_polygonmesh_bounding_box(m, std::ptr::null_mut()) });
        unsafe { truck_polygonmesh_free(m) };
    }

    #[test]
    fn free_null_is_safe() {
        // SAFETY: NULL is explicitly allowed.
        unsafe { truck_polygonmesh_free(std::ptr::null_mut()) };
    }
}

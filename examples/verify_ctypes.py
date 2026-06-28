#!/usr/bin/env python3
"""End-to-end verification of truck-bridge via ctypes.

Runs the release-built `truck_bridge.dll` through its real C ABI surface — no
Rust test harness involved. Covers stage 2 (foundation smoke) and stage 3
(OBJ/STL IO, to_buffer, merge, error path).

Run from the repo root after `cargo build --release`:
    python examples/verify_ctypes.py
"""

import ctypes
import math
import sys
from ctypes import (
    CDLL,
    Structure,
    c_bool,
    c_double,
    c_float,
    c_int32,
    c_size_t,
    c_uint32,
    c_uint8,
    c_void_p,
    byref,
    POINTER,
    cast,
)
from pathlib import Path


def as_u8_ptr(data: bytes):
    """Wrap a bytes object as a `POINTER(c_uint8)` for FFI byte-slice args."""
    n = len(data)
    arr = (c_uint8 * n).from_buffer_copy(data)
    return cast(arr, POINTER(c_uint8)), arr  # keep arr alive across the call


def as_f64_ptr(data):
    """Wrap a list/array of floats as a `POINTER(c_double)` for FFI args."""
    arr = (c_double * len(data))(*data)
    return cast(arr, POINTER(c_double)), arr


# --- C struct mirrors (must match include/truck_bridge.h) -------------------

class TruckF64Array(Structure):
    _fields_ = [("ptr", POINTER(c_double)), ("len", c_size_t)]

    def values(self):
        if not self.ptr or self.len == 0:
            return []
        return [self.ptr[i] for i in range(self.len)]


class TruckF32Array(Structure):
    _fields_ = [("ptr", POINTER(c_float)), ("len", c_size_t)]

    def values(self):
        if not self.ptr or self.len == 0:
            return []
        return [self.ptr[i] for i in range(self.len)]


class TruckU8Array(Structure):
    _fields_ = [("ptr", POINTER(c_uint8)), ("len", c_size_t)]

    def bytes(self):
        if not self.ptr or self.len == 0:
            return b""
        return bytes((self.ptr[i]) for i in range(self.len))


class TruckU32Array(Structure):
    _fields_ = [("ptr", POINTER(c_uint32)), ("len", c_size_t)]

    def values(self):
        if not self.ptr or self.len == 0:
            return []
        return [self.ptr[i] for i in range(self.len)]


class TruckStr(Structure):
    _fields_ = [("ptr", POINTER(c_uint8)), ("len", c_size_t)]

    def decode(self):
        if not self.ptr or self.len == 0:
            return ""
        raw = bytes((self.ptr[i]) for i in range(self.len))
        return raw.decode("utf-8")


# TruckStlType enum: Automatic=0, Ascii=1, Binary=2
STL_AUTOMATIC = 0
STL_ASCII = 1
STL_BINARY = 2


class TruckPolygonBuffer(Structure):
    _fields_ = [
        ("positions", TruckF64Array),
        ("uv", TruckF32Array),
        ("normal", TruckF32Array),
        ("indices", TruckU32Array),
    ]


# Array of opaque edge handles: { ptr: *mut *mut TruckEdge, len }
class TruckEdgeArray(Structure):
    _fields_ = [("ptr", POINTER(c_void_p)), ("len", c_size_t)]

    def handles(self):
        if not self.ptr or self.len == 0:
            return []
        return [self.ptr[i] for i in range(self.len)]


# A minimal one-triangle OBJ.
TRI_OBJ = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1 2 3\n"


def setup(lib):
    """Declare signatures on the loaded CDLL."""
    lib.truck_abi_version.restype = c_uint32

    lib.truck_polygonmesh_new_empty.restype = c_void_p
    lib.truck_polygonmesh_bounding_box.restype = c_bool
    lib.truck_polygonmesh_bounding_box.argtypes = [c_void_p, POINTER(TruckF64Array)]
    lib.truck_polygonmesh_free.argtypes = [c_void_p]

    lib.truck_polygonmesh_from_obj.restype = c_bool
    lib.truck_polygonmesh_from_obj.argtypes = [
        POINTER(c_uint8), c_size_t,
        POINTER(c_void_p), POINTER(c_void_p),
    ]
    lib.truck_polygonmesh_from_stl.restype = c_bool
    lib.truck_polygonmesh_from_stl.argtypes = [
        POINTER(c_uint8), c_size_t, c_uint32,
        POINTER(c_void_p), POINTER(c_void_p),
    ]
    lib.truck_polygonmesh_to_obj.restype = c_bool
    lib.truck_polygonmesh_to_obj.argtypes = [
        c_void_p, POINTER(TruckU8Array), POINTER(c_void_p),
    ]
    lib.truck_polygonmesh_to_stl.restype = c_bool
    lib.truck_polygonmesh_to_stl.argtypes = [
        c_void_p, c_uint32, POINTER(TruckU8Array), POINTER(c_void_p),
    ]
    lib.truck_polygonmesh_to_buffer.restype = c_bool
    lib.truck_polygonmesh_to_buffer.argtypes = [
        c_void_p, POINTER(TruckPolygonBuffer), POINTER(c_void_p),
    ]
    lib.truck_polygonmesh_merge.restype = c_bool
    lib.truck_polygonmesh_merge.argtypes = [
        c_void_p, c_void_p, POINTER(c_void_p),
    ]

    lib.truck_error_message.restype = TruckStr
    lib.truck_error_message.argtypes = [c_void_p]
    lib.truck_error_free.argtypes = [c_void_p]

    lib.truck_version_string.restype = TruckStr
    lib.truck_f64array_free.argtypes = [TruckF64Array]
    lib.truck_u8array_free.argtypes = [TruckU8Array]
    lib.truck_polygonbuffer_free.argtypes = [TruckPolygonBuffer]
    lib.truck_str_free.argtypes = [TruckStr]

    # stage 4a — topology Vertex
    lib.truck_vertex_new.restype = c_void_p
    lib.truck_vertex_new.argtypes = [c_double, c_double, c_double]
    lib.truck_vertex_point.restype = c_bool
    lib.truck_vertex_point.argtypes = [c_void_p, POINTER(TruckF64Array)]
    lib.truck_vertex_free.argtypes = [c_void_p]

    # stage 4b — topology Edge
    lib.truck_edge_line.restype = c_void_p
    lib.truck_edge_line.argtypes = [c_void_p, c_void_p]
    lib.truck_edge_circle_arc_by_transit.restype = c_void_p
    lib.truck_edge_circle_arc_by_transit.argtypes = [c_void_p, c_void_p, POINTER(c_double), c_size_t]
    lib.truck_edge_bezier.restype = c_void_p
    lib.truck_edge_bezier.argtypes = [c_void_p, c_void_p, POINTER(c_double), c_size_t]
    lib.truck_edge_front_vertex.restype = c_void_p
    lib.truck_edge_front_vertex.argtypes = [c_void_p]
    lib.truck_edge_back_vertex.restype = c_void_p
    lib.truck_edge_back_vertex.argtypes = [c_void_p]
    lib.truck_edge_free.argtypes = [c_void_p]

    # stage 4c — topology Face + TruckEdgeArray
    lib.truck_face_homotopy.restype = c_void_p
    lib.truck_face_homotopy.argtypes = [c_void_p, c_void_p]
    lib.truck_face_boundary_edge_count.restype = c_size_t
    lib.truck_face_boundary_edge_count.argtypes = [c_void_p]
    lib.truck_face_boundary_edges.restype = c_bool
    lib.truck_face_boundary_edges.argtypes = [c_void_p, POINTER(TruckEdgeArray)]
    lib.truck_face_free.argtypes = [c_void_p]
    lib.truck_edgearray_free.argtypes = [TruckEdgeArray]
    lib.truck_edgearray_free_all.argtypes = [TruckEdgeArray]


def get_error(lib, err_ptr):
    """If err_ptr is non-NULL, fetch + free the message, return it."""
    if not err_ptr:
        return None
    s = lib.truck_error_message(err_ptr)
    msg = s.decode()
    lib.truck_str_free(s)
    lib.truck_error_free(err_ptr)
    return msg


def main() -> int:
    dll_path = Path("target/release/truck_bridge.dll")
    if not dll_path.exists():
        print(f"ERROR: {dll_path} not found — run `cargo build --release` first.")
        return 1

    lib = CDLL(str(dll_path))
    setup(lib)

    # --- stage 2: ABI version + version string -----------------------------
    abi = lib.truck_abi_version()
    print(f"[1] ABI version: {abi}")
    assert abi == 1
    vs = lib.truck_version_string()
    vstr = vs.decode()
    print(f"    version string: {vstr}")
    assert "truck-bridge" in vstr
    lib.truck_str_free(vs)

    # --- stage 2: empty mesh bounding box ----------------------------------
    mesh = lib.truck_polygonmesh_new_empty()
    assert mesh
    arr = TruckF64Array()
    assert lib.truck_polygonmesh_bounding_box(mesh, byref(arr))
    bbox = arr.values()
    print(f"[2] empty bbox: {bbox}")
    assert len(bbox) == 6 and math.isinf(bbox[0]) and bbox[0] > 0
    lib.truck_f64array_free(arr)
    lib.truck_polygonmesh_free(mesh)

    # --- stage 3: from_obj --------------------------------------------------
    out = c_void_p()
    err = c_void_p()
    tri_ptr, _keep1 = as_u8_ptr(TRI_OBJ)
    ok = lib.truck_polygonmesh_from_obj(tri_ptr, len(TRI_OBJ), byref(out), byref(err))
    assert ok and out, f"from_obj failed: {get_error(lib, err)}"
    print(f"[3] from_obj: mesh handle {hex(out.value)}, 1 triangle parsed")
    m1 = out.value

    # --- stage 3: to_obj roundtrip -----------------------------------------
    obj_bytes = TruckU8Array()
    err = c_void_p()
    ok = lib.truck_polygonmesh_to_obj(m1, byref(obj_bytes), byref(err))
    assert ok and obj_bytes.len > 0, f"to_obj failed: {get_error(lib, err)}"
    print(f"[4] to_obj roundtrip: {obj_bytes.len} bytes")
    lib.truck_u8array_free(obj_bytes)

    # --- stage 3: to_buffer (separated arrays) -----------------------------
    buf = TruckPolygonBuffer()
    err = c_void_p()
    ok = lib.truck_polygonmesh_to_buffer(m1, byref(buf), byref(err))
    assert ok, f"to_buffer failed: {get_error(lib, err)}"
    assert buf.positions.len == 9, f"expected 9 position floats, got {buf.positions.len}"
    assert buf.indices.len == 3, f"expected 3 indices, got {buf.indices.len}"
    print(f"[5] to_buffer: {buf.positions.len // 3} verts, "
          f"{buf.indices.len // 3} triangle")
    print(f"    positions: {buf.positions.values()}")
    print(f"    indices:   {buf.indices.values()}")
    lib.truck_polygonbuffer_free(buf)

    # --- stage 3: merge -----------------------------------------------------
    out2 = c_void_p()
    err = c_void_p()
    tri_ptr2, _keep2 = as_u8_ptr(TRI_OBJ)
    ok = lib.truck_polygonmesh_from_obj(tri_ptr2, len(TRI_OBJ), byref(out2), byref(err))
    assert ok and out2
    err = c_void_p()
    ok = lib.truck_polygonmesh_merge(m1, out2.value, byref(err))
    assert ok, f"merge failed: {get_error(lib, err)}"
    # merged mesh should now have 2 triangles
    buf2 = TruckPolygonBuffer()
    err = c_void_p()
    ok = lib.truck_polygonmesh_to_buffer(m1, byref(buf2), byref(err))
    assert ok and buf2.indices.len == 6, f"expected 6 indices, got {buf2.indices.len}"
    print(f"[6] merge: combined mesh has {buf2.indices.len // 3} triangles")
    lib.truck_polygonbuffer_free(buf2)
    lib.truck_polygonmesh_free(m1)
    # out2 was consumed by merge; do NOT free it again

    # --- stage 3: error path (truncated binary STL) ------------------------
    out3 = c_void_p()
    err = c_void_p()
    truncated = b"\x00\x00\x00\x00truncated"
    tr_ptr, _keep3 = as_u8_ptr(truncated)
    ok = lib.truck_polygonmesh_from_stl(
        tr_ptr, len(truncated), STL_BINARY, byref(out3), byref(err))
    assert not ok, "truncated STL should fail"
    assert not out3.value
    msg = get_error(lib, err.value)
    assert msg, "expected a non-empty error message"
    print(f"[7] error path: truncated STL -> '{msg[:50]}...'")

    # --- stage 3: STL binary roundtrip -------------------------------------
    out4 = c_void_p()
    err = c_void_p()
    tri_ptr4, _keep4 = as_u8_ptr(TRI_OBJ)
    ok = lib.truck_polygonmesh_from_obj(tri_ptr4, len(TRI_OBJ), byref(out4), byref(err))
    assert ok and out4
    stl_bytes = TruckU8Array()
    err = c_void_p()
    ok = lib.truck_polygonmesh_to_stl(out4.value, STL_BINARY, byref(stl_bytes), byref(err))
    assert ok and stl_bytes.len > 0
    # re-read the stl bytes
    out5 = c_void_p()
    err = c_void_p()
    stl_data = stl_bytes.bytes()
    stl_ptr, _keep5 = as_u8_ptr(stl_data)
    ok = lib.truck_polygonmesh_from_stl(
        stl_ptr, len(stl_data), STL_BINARY, byref(out5), byref(err))
    assert ok and out5, f"stl roundtrip read failed: {get_error(lib, err)}"
    print(f"[8] STL binary roundtrip: wrote {stl_bytes.len} bytes, re-parsed OK")
    lib.truck_u8array_free(stl_bytes)
    lib.truck_polygonmesh_free(out4.value)
    lib.truck_polygonmesh_free(out5.value)

    # --- stage 4a: topology Vertex -----------------------------------------
    v = lib.truck_vertex_new(1.0, 2.0, 3.0)
    assert v, "vertex_new returned NULL"
    varr = TruckF64Array()
    assert lib.truck_vertex_point(v, byref(varr))
    pts = varr.values()
    assert pts == [1.0, 2.0, 3.0], f"vertex point mismatch: {pts}"
    print(f"[9] vertex new/point: {pts}")
    lib.truck_f64array_free(varr)
    lib.truck_vertex_free(v)
    # NULL safety
    assert not lib.truck_vertex_point(None, byref(TruckF64Array()))
    lib.truck_vertex_free(None)  # idempotent

    # --- stage 4b: topology Edge -------------------------------------------
    va = lib.truck_vertex_new(1.0, 0.0, 0.0)
    vb = lib.truck_vertex_new(-1.0, 0.0, 0.0)

    # line + front/back endpoint roundtrip
    e_line = lib.truck_edge_line(va, vb)
    assert e_line, "edge_line returned NULL"
    fv = lib.truck_edge_front_vertex(e_line)
    bv = lib.truck_edge_back_vertex(e_line)
    assert fv and bv
    farr = TruckF64Array()
    barr = TruckF64Array()
    assert lib.truck_vertex_point(fv, byref(farr))
    assert lib.truck_vertex_point(bv, byref(barr))
    assert farr.values() == [1.0, 0.0, 0.0], f"front vertex mismatch: {farr.values()}"
    assert barr.values() == [-1.0, 0.0, 0.0], f"back vertex mismatch: {barr.values()}"
    print(f"[10] edge line: front={farr.values()}, back={barr.values()}")
    lib.truck_f64array_free(farr)
    lib.truck_f64array_free(barr)
    lib.truck_vertex_free(fv)
    lib.truck_vertex_free(bv)
    lib.truck_edge_free(e_line)

    # circle_arc by transit (upper semicircle through (0,1,0))
    transit_ptr, _k = as_f64_ptr([0.0, 1.0, 0.0])
    e_arc = lib.truck_edge_circle_arc_by_transit(va, vb, transit_ptr, 3)
    assert e_arc, "edge_circle_arc_by_transit returned NULL"
    print("[11] edge circle_arc by transit: OK")
    lib.truck_edge_free(e_arc)

    # circle_arc NULL transit -> NULL
    assert not lib.truck_edge_circle_arc_by_transit(va, vb, None, 0)

    # bezier with 2 control points
    ctrl_ptr, _k2 = as_f64_ptr([1.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    e_bez = lib.truck_edge_bezier(va, vb, ctrl_ptr, 6)
    assert e_bez, "edge_bezier returned NULL"
    print("[12] edge bezier: OK")
    lib.truck_edge_free(e_bez)

    # bezier bad length (2 floats, not multiple of 3) -> NULL
    bad_ptr, _k3 = as_f64_ptr([1.0, 2.0])
    assert not lib.truck_edge_bezier(va, vb, bad_ptr, 2), "bezier bad length should be NULL"

    # NULL vertex -> line NULL
    assert not lib.truck_edge_line(None, vb)
    lib.truck_vertex_free(va)
    lib.truck_vertex_free(vb)

    # --- stage 4c: topology Face + TruckEdgeArray --------------------------
    # Build two parallel edges, then a homotopy face between them.
    f_v0 = lib.truck_vertex_new(0.0, 0.0, 0.0)
    f_v1 = lib.truck_vertex_new(1.0, 0.0, 0.0)
    f_v2 = lib.truck_vertex_new(0.0, 0.0, 1.0)
    f_v3 = lib.truck_vertex_new(1.0, 0.0, 1.0)
    fe0 = lib.truck_edge_line(f_v0, f_v1)
    fe1 = lib.truck_edge_line(f_v2, f_v3)
    assert fe0 and fe1
    face = lib.truck_face_homotopy(fe0, fe1)
    assert face, "face_homotopy returned NULL"

    count = lib.truck_face_boundary_edge_count(face)
    assert count == 4, f"homotopy face should have 4 boundary edges, got {count}"
    print(f"[13] face homotopy: {count} boundary edges")

    # boundary edges enumeration
    earr = TruckEdgeArray()
    assert lib.truck_face_boundary_edges(face, byref(earr))
    assert earr.len == count, f"edge array len {earr.len} != count {count}"
    hs = earr.handles()
    assert all(h for h in hs), "all boundary edge handles must be non-null"
    print(f"[14] face boundary_edges: {earr.len} independent handles")
    # free_all releases container + all handles
    lib.truck_edgearray_free_all(earr)

    # NULL safety
    assert not lib.truck_face_boundary_edges(None, byref(TruckEdgeArray()))
    assert not lib.truck_face_homotopy(None, fe1)
    lib.truck_face_free(face)
    lib.truck_edge_free(fe0)
    lib.truck_edge_free(fe1)
    lib.truck_vertex_free(f_v0)
    lib.truck_vertex_free(f_v1)
    lib.truck_vertex_free(f_v2)
    lib.truck_vertex_free(f_v3)

    print("\nAll ctypes checks passed (stage 2 + 3 + 4a + 4b + 4c).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

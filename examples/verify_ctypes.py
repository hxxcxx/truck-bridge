#!/usr/bin/env python3
"""End-to-end verification of truck-bridge via ctypes.

Runs the release-built `truck_bridge.dll` through its real C ABI surface — no
Rust test harness involved. This proves the four foundations hold against the
actual artifact a C/C++ consumer would link.

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
    c_size_t,
    c_uint32,
    c_uint8,
    c_void_p,
    byref,
    POINTER,
)
from pathlib import Path


# --- C struct mirrors (must match truck_bridge.h) ---------------------------

class TruckF64Array(Structure):
    _fields_ = [("ptr", POINTER(c_double)), ("len", c_size_t)]

    def values(self):
        if not self.ptr or self.len == 0:
            return []
        return [self.ptr[i] for i in range(self.len)]


class TruckStr(Structure):
    _fields_ = [("ptr", POINTER(c_uint8)), ("len", c_size_t)]

    def decode(self):
        if not self.ptr or self.len == 0:
            return ""
        raw = bytes(self.ptr[i] for i in range(self.len))
        return raw.decode("utf-8")


def main() -> int:
    dll_path = Path("target/release/truck_bridge.dll")
    if not dll_path.exists():
        print(f"ERROR: {dll_path} not found — run `cargo build --release` first.")
        return 1

    lib = CDLL(str(dll_path))

    # --- signatures ---------------------------------------------------------
    lib.truck_abi_version.restype = c_uint32
    lib.truck_polygonmesh_new_empty.restype = c_void_p
    lib.truck_polygonmesh_bounding_box.restype = c_bool
    lib.truck_polygonmesh_bounding_box.argtypes = [c_void_p, POINTER(TruckF64Array)]
    lib.truck_polygonmesh_free.argtypes = [c_void_p]
    lib.truck_version_string.restype = TruckStr
    lib.truck_f64array_free.argtypes = [TruckF64Array]
    lib.truck_str_free.argtypes = [TruckStr]

    # --- 1. ABI version -----------------------------------------------------
    abi = lib.truck_abi_version()
    print(f"ABI version: {abi}")
    assert abi == 1, "ABI version mismatch"

    # --- 2. version string --------------------------------------------------
    vs = lib.truck_version_string()
    vstr = vs.decode()
    print(f"version string: {vstr}")
    assert "truck-bridge" in vstr
    lib.truck_str_free(vs)

    # --- 3. empty mesh bounding box ----------------------------------------
    mesh = lib.truck_polygonmesh_new_empty()
    assert mesh, "new_empty returned NULL"
    print(f"mesh handle: {hex(mesh)}")

    arr = TruckF64Array()
    ok = lib.truck_polygonmesh_bounding_box(mesh, byref(arr))
    assert ok, "bounding_box returned false"
    bbox = arr.values()
    print(f"empty bbox: {bbox}")
    # empty mesh => min = +INF, max = -INF
    assert len(bbox) == 6
    assert math.isinf(bbox[0]) and bbox[0] > 0, "min.x should be +INF"
    assert math.isinf(bbox[3]) and bbox[3] < 0, "max.x should be -INF"
    lib.truck_f64array_free(arr)

    # --- 4. free (idempotent, NULL-safe) -----------------------------------
    lib.truck_polygonmesh_free(mesh)
    lib.truck_polygonmesh_free(None)  # NULL must be tolerated

    # --- 5. NULL mesh -> bounding_box false (no crash) ---------------------
    arr2 = TruckF64Array()
    assert not lib.truck_polygonmesh_bounding_box(None, byref(arr2)), \
        "NULL mesh should yield false"

    print("\nAll ctypes checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Validate that the C++ lookup tables return identical values to the original
numpy lambda tables.

Usage:
    python test_lookup_tables.py <path_to_test_lookup_tables_exe>

The executable is built from src/test_lookup_tables.cpp via CMake.
"""
import sys
import subprocess
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from ob_table_array import table_one as ob_t1, table_two as ob_t2
from tb_table_array import (table_one as tb_t1, table_two as tb_t2,
                             table_three as tb_t3, table_four as tb_t4)

BS = [2.0, 3.0, 5.0]
TOL = 1e-12


def python_values():
    """Generate the same key->value mapping that the C++ exe prints."""
    vals = {}
    for bi, b in enumerate(BS):
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    vals[f"ob_table_one,{bi},{i},{j},{k}"] = ob_t1[i,j,k](b)
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    for l in range(2):
                        vals[f"ob_table_two,{bi},{i},{j},{k},{l}"] = ob_t2[i,j,k,l](b)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(2):
                        vals[f"tb_table_one,{bi},{i},{j},{k},{l}"] = tb_t1[i,j,k,l](b)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(2):
                        vals[f"tb_table_two,{bi},{i},{j},{k},{l}"] = tb_t2[i,j,k,l](b)
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    for l in range(2):
                        for m in range(2):
                            vals[f"tb_table_three,{bi},{i},{j},{k},{l},{m}"] = tb_t3[i,j,k,l,m](b)
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    for l in range(3):
                        for m in range(2):
                            vals[f"tb_table_four,{bi},{i},{j},{k},{l},{m}"] = tb_t4[i,j,k,l,m](b)
    return vals


def cpp_values(exe):
    result = subprocess.run([exe], capture_output=True, text=True, check=True)
    vals = {}
    for line in result.stdout.splitlines():
        key, _, raw = line.rpartition(",")
        vals[key] = float(raw)
    return vals


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_lookup_tables.py <path_to_test_lookup_tables_exe>")
        sys.exit(1)

    exe = sys.argv[1]

    print("Generating Python reference values ...")
    py_vals = python_values()

    print(f"Running C++ executable: {exe}")
    cpp_vals = cpp_values(exe)

    # Check key sets match
    missing = set(py_vals) - set(cpp_vals)
    extra   = set(cpp_vals) - set(py_vals)
    if missing or extra:
        if missing:
            print(f"  Keys missing from C++ ({len(missing)}): {sorted(missing)[:5]}")
        if extra:
            print(f"  Extra keys in C++ ({len(extra)}): {sorted(extra)[:5]}")
        print("FAIL: key sets differ")
        sys.exit(1)

    failures = [
        (key, py_vals[key], cpp_vals[key], abs(py_vals[key] - cpp_vals[key]))
        for key in py_vals
        if abs(py_vals[key] - cpp_vals[key]) > TOL
    ]

    n = len(py_vals)
    if failures:
        print(f"\nFAIL: {len(failures)}/{n} entries exceed tolerance {TOL}:")
        for key, py_v, cpp_v, diff in failures[:20]:
            print(f"  {key}")
            print(f"    python={py_v:.15g}  cpp={cpp_v:.15g}  diff={diff:.3e}")
        sys.exit(1)
    else:
        print(f"PASS: all {n} entries match (tol={TOL})")


if __name__ == "__main__":
    main()

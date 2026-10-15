#!/usr/bin/env python3
"""Per-op compute-unit report (ANE / GPU / CPU) for a CoreML model (macOS only)

Uses the coremltools MLComputePlan API to report, for each operation of an .mlpackage, the
compute device Core ML prefers (`preferred_compute_device`) and its estimated cost:
- op count and cost share per device
- op types by cost, with the devices they are assigned to
- optionally every op in execution order (--all)

Requirements: macOS 14.4+ and coremltools 8.0+ (MLComputePlan calls the Core ML framework).

The exported RTMDet-Fusion models are meant to run with CPU_AND_GPU; the Neural Engine produces
broken numerics for them.

Usage:
    uv run python tools/coreml_compute_plan.py --mlpackage rtmdet_fusion_coreml_s_640_v2.0.0.mlpackage
    # Compute units Core ML may use (default ALL)
    uv run python tools/coreml_compute_plan.py --mlpackage m.mlpackage --compute-units CPU_AND_GPU
    # List every op
    uv run python tools/coreml_compute_plan.py --mlpackage m.mlpackage --all
"""

import argparse
from collections import defaultdict


_DEVICE_NAME = {
    "MLCPUComputeDevice": "CPU",
    "MLGPUComputeDevice": "GPU",
    "MLNeuralEngineComputeDevice": "ANE",
}


def device_label(dev) -> str:
    if dev is None:
        return "?"
    return _DEVICE_NAME.get(type(dev).__name__, type(dev).__name__)


def _iter_operations(block):
    """Walk block.operations recursively (including nested blocks of control-flow ops)."""
    for op in block.operations:
        yield op
        for sub in getattr(op, "blocks", None) or []:
            yield from _iter_operations(sub)


def main() -> None:
    ap = argparse.ArgumentParser(description="CoreML per-op compute-unit report (macOS only)")
    ap.add_argument("--mlpackage", required=True, help="Path of the .mlpackage (or .mlmodelc) to analyze")
    ap.add_argument("--compute-units", default="ALL",
                    choices=["ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"],
                    help="Compute units Core ML may use (default: ALL)")
    ap.add_argument("--function", default="main", help="Program function to analyze (default: main)")
    ap.add_argument("--all", action="store_true", help="List every op (default: summary only)")
    args = ap.parse_args()

    import coremltools as ct
    try:
        from coremltools.models.compute_plan import MLComputePlan
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Cannot import MLComputePlan (requires macOS 14.4+ / coremltools>=8.0): {e}"
        )

    compute_units = getattr(ct.ComputeUnit, args.compute_units)

    # MLComputePlan.load_from_path needs a compiled model (.mlmodelc). Passing an .mlpackage
    # aborts in native code (libc++abi terminate, not a Python exception), so compile first.
    path = args.mlpackage.rstrip("/")
    if path.endswith(".mlmodelc"):
        compiled_path = path
    else:
        print(f"Compiling {path} -> .mlmodelc ...")
        try:
            from coremltools.models.utils import compile_model
            compiled_path = compile_model(path)
        except Exception as e:  # noqa: BLE001
            # Fallback for older coremltools
            print(f"  compile_model failed ({e}); trying MLModel.get_compiled_model_path()")
            compiled_path = ct.models.MLModel(
                path, compute_units=compute_units
            ).get_compiled_model_path()
    print(f"compiled: {compiled_path}")

    plan = MLComputePlan.load_from_path(compiled_path, compute_units=compute_units)

    program = plan.model_structure.program
    if program is None:
        raise SystemExit(
            "No ML Program structure found (the model may be a NeuralNetwork). "
            "This tool supports ML Program (mlprogram) models only."
        )

    functions = program.functions
    func = functions.get(args.function) or next(iter(functions.values()))
    ops = list(_iter_operations(func.block))

    per_device_cost = defaultdict(float)
    per_device_count = defaultdict(int)
    per_type = defaultdict(lambda: [0, 0.0, set()])  # op_type -> [count, cost, devices]
    rows = []

    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        cost_obj = plan.get_estimated_cost_for_mlprogram_operation(op)
        dev = device_label(getattr(usage, "preferred_compute_device", None)) if usage else "?"
        cost = float(getattr(cost_obj, "weight", 0.0)) if cost_obj else 0.0
        op_type = getattr(op, "operator_name", None) or getattr(op, "op_type", "?")

        per_device_cost[dev] += cost
        per_device_count[dev] += 1
        per_type[op_type][0] += 1
        per_type[op_type][1] += cost
        per_type[op_type][2].add(dev)
        rows.append((op_type, dev, cost))

    total_cost = sum(per_device_cost.values()) or 1.0

    print("\n=== CoreML Compute Plan ===")
    print(f"model        : {path}")
    print(f"compute_units: {args.compute_units}")
    print(f"function     : {func and getattr(func, 'name', args.function)}")
    print(f"total ops    : {len(ops)}\n")

    print("device    ops     cost%")
    for d in ("ANE", "GPU", "CPU", "?"):
        if per_device_count.get(d):
            print(f"  {d:<6}{per_device_count[d]:>6}  {100 * per_device_cost[d] / total_cost:6.1f}%")

    print("\n--- op types by cost ---")
    for op_type, (cnt, cost, devs) in sorted(per_type.items(), key=lambda x: -x[1][1]):
        print(f"  {op_type:<30} x{cnt:<5} {100 * cost / total_cost:5.1f}%  {'/'.join(sorted(devs))}")

    if args.all:
        print("\n--- all ops (execution order) ---")
        for i, (op_type, dev, cost) in enumerate(rows):
            print(f"  [{i:4}] {op_type:<30} {dev:<4} {100 * cost / total_cost:5.2f}%")


if __name__ == "__main__":
    main()

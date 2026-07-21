#!/usr/bin/env python3
"""Validate a gameplay pipeline-impact report against a versioned contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT = (
    Path(__file__).parents[1] / "validation" / "gameplay_gate_eval10_v2.json"
)


def nested_value(payload: dict[str, Any], path: str) -> Any:
    """Resolve a dot-separated path in a JSON object."""
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"Missing report path: {path}")
        value = value[component]
    return value


def scalar_check(
    observed: float,
    operator: str,
    expected: float,
    tolerance: float,
) -> bool:
    """Apply one numeric comparison with an explicit tolerance."""
    if not math.isfinite(observed) or not math.isfinite(expected):
        return False
    if operator == "eq":
        return abs(observed - expected) <= tolerance
    if operator == "min":
        return observed + tolerance >= expected
    if operator == "max":
        return observed <= expected + tolerance
    raise ValueError(f"Unsupported numeric operator: {operator}")


def evaluate_population_check(
    report: dict[str, Any],
    check: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one exact population/provenance check."""
    raw_observed = nested_value(report, str(check["path"]))
    operator = str(check["operator"])
    expected = check["value"]
    if operator == "length_eq":
        if not hasattr(raw_observed, "__len__"):
            raise TypeError(f"Population path is not sized: {check['path']}")
        observed = len(raw_observed)
        passed = observed == int(expected)
    elif operator == "eq":
        observed = raw_observed
        passed = observed == expected
    else:
        raise ValueError(f"Unsupported population operator: {operator}")
    return {
        "id": str(check["id"]),
        "path": str(check["path"]),
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def evaluate_metric_check(
    report: dict[str, Any],
    check: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one gameplay validation metric."""
    kind = str(check["kind"])
    operator = str(check["operator"])
    expected = float(check["value"])
    tolerance = float(check.get("tolerance", 0.0))
    result: dict[str, Any] = {
        "id": str(check["id"]),
        "label": str(check["label"]),
        "kind": kind,
        "operator": operator,
        "expected": expected,
        "tolerance": tolerance,
    }
    if kind == "scalar":
        observed = float(nested_value(report, str(check["path"])))
        passed = scalar_check(observed, operator, expected, tolerance)
        result.update(
            {
                "path": str(check["path"]),
                "observed": observed,
                "passed": passed,
            }
        )
        return result
    if kind == "ratio":
        numerator = int(nested_value(report, str(check["numerator_path"])))
        denominator = int(nested_value(report, str(check["denominator_path"])))
        ratio = numerator / denominator if denominator else 0.0
        minimum_numerator = int(check.get("minimum_numerator", 0))
        passed = scalar_check(ratio, operator, expected, tolerance)
        passed = passed and numerator >= minimum_numerator
        result.update(
            {
                "numerator_path": str(check["numerator_path"]),
                "denominator_path": str(check["denominator_path"]),
                "observed_numerator": numerator,
                "observed_denominator": denominator,
                "observed": ratio,
                "minimum_numerator": minimum_numerator,
                "passed": passed,
            }
        )
        return result
    raise ValueError(f"Unsupported metric kind: {kind}")


def evaluate_contract(
    report: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate population identity and the gameplay gate metrics."""
    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported gameplay validation contract schema")
    population_results = [
        evaluate_population_check(report, check)
        for check in contract.get("population_checks", [])
    ]
    metric_results = [
        evaluate_metric_check(report, check)
        for check in contract.get("metric_checks", [])
    ]
    failed_population = [row["id"] for row in population_results if not row["passed"]]
    failed_metrics = [row["id"] for row in metric_results if not row["passed"]]
    if failed_population:
        status = "invalid_population"
    elif failed_metrics:
        status = "fail"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "contract_id": contract.get("contract_id"),
        "status": status,
        "population_checks": population_results,
        "metric_checks": metric_results,
        "failed_population_ids": failed_population,
        "failed_metric_ids": failed_metrics,
        "summary": {
            "metric_checks": len(metric_results),
            "passed": len(metric_results) - len(failed_metrics),
            "failed": len(failed_metrics),
        },
    }


def markdown_result(result: dict[str, Any]) -> str:
    """Render a compact validation-gate report."""
    lines = [
        "# Gameplay Validation Gate",
        "",
        f"Status: **{str(result['status']).upper()}**",
        "",
        "| Metric | Observed | Requirement | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in result["metric_checks"]:
        if row["kind"] == "ratio":
            observed = (
                f"{row['observed_numerator']}/{row['observed_denominator']} "
                f"({100 * float(row['observed']):.4f}%)"
            )
            requirement = (
                f">= {row['minimum_numerator']} and "
                f">= {100 * float(row['expected']):.4f}%"
            )
        else:
            observed = f"{float(row['observed']):.10g}"
            symbol = {"min": ">=", "max": "<=", "eq": "=="}[row["operator"]]
            requirement = f"{symbol} {float(row['expected']):.10g}"
        lines.append(
            f"| {row['label']} | {observed} | {requirement} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    if result["failed_population_ids"]:
        lines.extend(
            [
                "",
                "Population mismatch: " + ", ".join(result["failed_population_ids"]),
                "",
                "Create a new versioned contract rather than comparing changed membership to this baseline.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the gameplay validation gate and return a CI-friendly exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = evaluate_contract(report, contract)
    result.update(
        {
            "contract": str(args.contract),
            "report": str(args.report),
        }
    )
    if args.output is not None:
        if args.output.exists() and not args.force:
            raise FileExistsError(f"Refusing existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        args.output.with_suffix(".md").write_text(
            markdown_result(result), encoding="utf-8"
        )
    print(markdown_result(result), end="")
    if result["status"] == "pass":
        return
    if result["status"] == "invalid_population":
        raise SystemExit(2)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

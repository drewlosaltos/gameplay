#!/usr/bin/env python3
"""Reproduce a SkyBall model from its recipe.

    python scripts/reproduce.py recipes/<model_id>.json [--fetch] [--check] [--train] [--evaluate] [--export]

With no stage flag it runs fetch, check, train and evaluate. Exit codes: 0 PASS/ok,
1 error, 2 FAIL. The recipe schema is
skyball_repro_recipe/v1, shared with the jersey and CLIP-ReID repos. See REPRODUCE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from skyball_gameplay.data_root import data_root, dataset_path, hf_snapshot_path  # noqa: E402

SCHEMA = "skyball_repro_recipe/v1"
REQUIRED = ("schema", "model_id", "task", "code", "env", "inputs", "hardware", "train", "select", "expected", "evaluate")
INPUT_FIELDS = {"hf": ("hf_repo", "revision", "allow_patterns", "identity"),
                "drive": ("remote", "local", "sha256"),
                "data_root": ("path", "fetch", "identity")}
SELECT_RULES = {"max_metric": max, "min_metric": min}
DEFAULT_STAGES = ("fetch", "check", "train", "evaluate")
TASKS = ("gameplay_classification",)


class RecipeError(ValueError):
    """The recipe does not follow skyball_repro_recipe/v1."""


# ---------------------------------------------------------------- recipe

def validate_recipe(recipe: dict) -> dict:
    missing = [key for key in REQUIRED if key not in recipe]
    if missing:
        raise RecipeError(f"Recipe is missing {missing}")
    if recipe["schema"] != SCHEMA:
        raise RecipeError(f"Unsupported schema {recipe['schema']!r}; expected {SCHEMA}")
    if recipe["task"] not in TASKS:
        raise RecipeError(f"Task {recipe['task']!r} has no launcher in this repo; supported: {TASKS}")
    for key in ("min_commit", "provenance_note"):
        if key not in recipe["code"]:
            raise RecipeError(f"code.{key} is required (null is allowed for min_commit)")
    for key in ("lockfile", "require", "pythonpath_overlays"):
        if key not in recipe["env"]:
            raise RecipeError(f"env.{key} is required")
    if "python" not in recipe["env"]["require"]:
        raise RecipeError("env.require must include python")
    for overlay in recipe["env"]["pythonpath_overlays"]:
        for key in ("name", "install_cmd", "path_env"):
            if key not in overlay:
                raise RecipeError(f"env.pythonpath_overlays entry lacks {key}")
    for name, spec in recipe["inputs"].items():
        kind = spec.get("kind")
        if kind not in INPUT_FIELDS:
            raise RecipeError(f"inputs.{name}.kind must be one of {sorted(INPUT_FIELDS)}")
        for key in INPUT_FIELDS[kind]:
            if key not in spec:
                raise RecipeError(f"inputs.{name} ({kind}) lacks {key}")
        if "method" not in spec.get("fetch", {}):
            raise RecipeError(f"inputs.{name}.fetch.method is required")
    for key in ("gpus", "gpu_type", "expected_wall_h"):
        if key not in recipe["hardware"]:
            raise RecipeError(f"hardware.{key} is required")
    select = recipe["select"]
    for key in ("rule", "metric"):
        if key not in select:
            raise RecipeError(f"select.{key} is required")
    if select["rule"] not in SELECT_RULES:
        raise RecipeError(f"select.rule must be one of {sorted(SELECT_RULES)}")
    expected = recipe["expected"]
    for key in ("per_epoch", "selected_epoch", "tolerance", "prod_reference"):
        if key not in expected:
            raise RecipeError(f"expected.{key} is required")
    if not expected["per_epoch"] or any("epoch" not in row or select["metric"] not in row
                                        for row in expected["per_epoch"]):
        raise RecipeError(f"expected.per_epoch rows need epoch and {select['metric']}")
    if f"{select['metric']}_abs" not in expected["tolerance"]:
        raise RecipeError(f"expected.tolerance.{select['metric']}_abs is required")
    if expected["selected_epoch"] not in {row["epoch"] for row in expected["per_epoch"]}:
        raise RecipeError("expected.selected_epoch is not one of expected.per_epoch")
    return recipe


def load_recipe(path: Path) -> dict:
    return validate_recipe(json.loads(Path(path).read_text()))


def load_config(recipe: dict) -> dict:
    return json.loads((REPO / recipe["train"]["config"]).read_text())


def config_value(config: dict, flag: str) -> str:
    return config["args"][config["args"].index(flag) + 1]


def recipe_config_mismatches(recipe: dict, config: dict) -> list[str]:
    """Everything the recipe states that the canonical config also states must agree."""
    problems = []
    if config["recipe_id"] != recipe["train"]["recipe_id"]:
        problems.append("recipe_id")
    if config["dataset"] != recipe["inputs"]["dataset"]["path"]:
        problems.append("dataset path")
    if config["split_contract"] != recipe["select"].get("split_contract"):
        problems.append("split contract")
    if config["gpu_count"] != recipe["hardware"]["gpus"]:
        problems.append("gpu count")
    if int(config_value(config, "--epochs")) != len(recipe["expected"]["per_epoch"]):
        problems.append("epochs vs expected.per_epoch")
    if config_value(config, "--selection-metric") != recipe["select"]["metric"]:
        problems.append("selection metric")
    base = recipe["inputs"]["qwen_base"]
    suffix = Path(f"models--{base['hf_repo'].replace('/', '--')}") / "snapshots" / base["revision"]
    if not config_value(config, "--qwen-snapshot").endswith(str(suffix)):
        problems.append("qwen snapshot repo/revision")
    if not recipe["train"]["checkpoint_name"].endswith(config_value(config, "--run-tag") + ".pth.tar"):
        problems.append("checkpoint name vs run tag")
    return problems


# ---------------------------------------------------------------- inputs

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def env_path(name: str | None, default: str | None) -> Path | None:
    value = os.environ.get(name) if name else None
    return Path(value) if value else (Path(default) if default else None)


def repro_cache() -> Path:
    """Downloaded checkpoints and prod references; never inside the checkout."""
    return Path(os.environ.get("SKYBALL_REPRO_CACHE") or Path.home() / ".cache/skyball_repro")


def staging_root() -> Path:
    """Where Drive zips are staged before extraction: <data_root>/../unified_staging by default."""
    return Path(os.environ.get("SKYBALL_STAGING_ROOT") or data_root().parent / "unified_staging")


def input_location(name: str, spec: dict, recipe: dict) -> Path:
    fetch = spec.get("fetch", {})
    if spec["kind"] == "hf":
        override = os.environ.get(fetch["path_env"]) if fetch.get("path_env") else None
        return Path(override) if override else hf_snapshot_path(spec["hf_repo"], spec["revision"])
    if spec["kind"] == "data_root":
        return dataset_path(spec["path"])
    return repro_cache() / spec["local"]


def input_stages(spec: dict) -> set[str]:
    return set(spec.get("stages", ("check", "train", "evaluate")))


def needed_inputs(recipe: dict, stages: set[str]) -> dict:
    """Inputs for check/train/evaluate, plus export-only inputs when --export is requested."""
    wanted = {"check", "train", "evaluate"} | (stages & {"export"})
    return {name: spec for name, spec in recipe["inputs"].items() if input_stages(spec) & wanted}


def fetch_commands(name: str, spec: dict, recipe: dict) -> list[str]:
    where = input_location(name, spec, recipe)
    fetch = spec["fetch"]
    q = shlex.quote
    if fetch["method"] == "hf_download":
        cmd = ["hf", "download", spec["hf_repo"], "--revision", spec["revision"], *spec["allow_patterns"]]
        if fetch.get("path_env") and os.environ.get(fetch["path_env"]):
            cmd += ["--local-dir", str(where)]
        return [" ".join(q(part) for part in cmd)]
    if fetch["method"] == "rclone_copy":
        excludes = " ".join(f"--exclude {q(pattern)}" for pattern in fetch.get("exclude", []))
        return [f"rclone copy {q(spec['remote'])} {q(str(where))} {excludes}".strip()]
    if fetch["method"] == "rclone_zip_shards":
        staging = staging_root() / spec["path"]
        target = dataset_path(fetch["unzip_into"])
        return [f"rclone copy {q(fetch['remote'])} {q(str(staging))}",
                f"cd {q(str(staging))} && sha256sum -c {q(fetch.get('checksums', 'SHA256SUMS'))}",
                f"mkdir -p {q(str(target))}",
                f"cd {q(str(staging))} && for z in *.zip; do unzip -q -n \"$z\" -d {q(str(target))}; done"]
    raise RecipeError(f"Unknown fetch.method {fetch['method']!r} for {name}")


def check_identity(name: str, spec: dict, recipe: dict) -> dict:
    where = input_location(name, spec, recipe)
    result = {"input": name, "path": str(where), "checks": []}

    def add(label, path, expected):
        found = sha256(path) if path.is_file() else None
        result["checks"].append({"file": label, "expected": expected, "found": found, "ok": found == expected})

    if spec["kind"] == "data_root":
        identity = spec["identity"]
        add("metadata/windows.jsonl", where / "metadata/windows.jsonl", identity["windows_sha256"])
        if "summary_sha256" in identity:
            add("metadata/summary.json", where / "metadata/summary.json", identity["summary_sha256"])
    else:
        hashes = spec["identity"]["sha256"] if spec["kind"] == "hf" else spec["sha256"]
        if isinstance(hashes, str):
            add(where.name, where, hashes)
        else:
            for label, expected in hashes.items():
                add(label, where / label, expected)
    result["ok"] = bool(result["checks"]) and all(check["ok"] for check in result["checks"])
    return result


# ---------------------------------------------------------------- environment

def version_matches(required: str, found: str | None) -> bool:
    if found is None:
        return False
    if "+" in required:
        return found == required
    return found.split("+")[0] == required or found.startswith(required + ".")


def activate_overlays(recipe: dict) -> list[dict]:
    rows = []
    for overlay in recipe["env"]["pythonpath_overlays"]:
        path = env_path(overlay["path_env"], overlay.get("default"))
        present = bool(path and path.is_dir())
        if present and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(path), os.environ.get("PYTHONPATH")]))
        rows.append({"overlay": overlay["name"], "path": str(path), "present": present,
                     "install_cmd": overlay["install_cmd"]})
    importlib.invalidate_caches()
    return rows


def installed_version(package: str) -> str | None:
    if package == "python":
        return ".".join(map(str, sys.version_info[:3]))
    for finder in sys.path:
        for dist in importlib.metadata.distributions(path=[finder]):
            if dist.metadata["Name"] and dist.metadata["Name"].lower() == package.lower():
                return dist.version
    return None


def check_env(recipe: dict) -> list[dict]:
    rows = []
    for package, required in recipe["env"]["require"].items():
        found = installed_version(package)
        rows.append({"package": package, "required": required, "found": found,
                     "ok": version_matches(required, found)})
    return rows


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain")),
            "describe": run("describe", "--always", "--dirty")}


def code_check(recipe: dict) -> dict:
    state = git_state()
    minimum = recipe["code"]["min_commit"]
    ok = True
    if minimum:
        ok = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", minimum, "HEAD"]).returncode == 0
    return {**state, "min_commit": minimum, "ok": ok}


# ---------------------------------------------------------------- evaluation

def read_epoch_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_epoch(rows: list[dict], select: dict) -> dict:
    if not rows:
        raise ValueError("No epoch metrics to select from")
    pick = SELECT_RULES[select["rule"]]
    sign = 1 if select["rule"] == "max_metric" else -1
    return pick(rows, key=lambda row: (row[select["metric"]], -sign * row["epoch"]))


def within(expected: float, found: float | None, tolerance: float) -> bool:
    return found is not None and abs(found - expected) <= tolerance


def check_row(name: str, status: str, observed=None, expected=None, tolerance=None) -> dict:
    return {"name": name, "status": status, "observed": observed, "expected": expected, "tolerance": tolerance}


def build_report(recipe: dict, rows: list[dict], rescore: dict | None = None) -> dict:
    """Reproduced = selected checkpoint's metric and the per-epoch curve within tolerance of prod.

    A different selected epoch is INFO. Full10 is a separate documented step and is not included.
    """
    metric = recipe["select"]["metric"]
    tolerance = recipe["expected"]["tolerance"]
    tol = tolerance[f"{metric}_abs"]
    checks = []
    by_epoch = {row["epoch"]: row for row in rows}
    for expected in recipe["expected"]["per_epoch"]:
        found = by_epoch.get(expected["epoch"], {})
        checks.append(check_row(f"epoch {expected['epoch']} {metric}",
                                "PASS" if within(expected[metric], found.get(metric), tol) else "FAIL",
                                found.get(metric), expected[metric], tol))
        if tolerance.get("count_exact") and "count" in expected:
            checks.append(check_row(f"epoch {expected['epoch']} count",
                                    "PASS" if found.get("count") == expected["count"] else "FAIL",
                                    found.get("count"), expected["count"], 0))
    selected = select_epoch(rows, recipe["select"]) if rows else None
    prod_epoch = recipe["expected"]["selected_epoch"]
    if selected is None:
        checks.append(check_row("selected epoch", "FAIL", None, prod_epoch))
    else:
        checks.append(check_row("selected epoch", "PASS" if selected["epoch"] == prod_epoch else "INFO",
                                selected["epoch"], prod_epoch))
    reference = next(row[metric] for row in recipe["expected"]["per_epoch"] if row["epoch"] == prod_epoch)
    checks.append(check_row(f"selected {metric}",
                            "PASS" if selected and within(reference, selected[metric], tol) else "FAIL",
                            selected and selected[metric], reference, tol))
    if rescore is None:
        checks.append(check_row(f"standalone re-score {metric}", "SKIPPED"))
    else:
        found = rescore.get(metric)
        checks.append(check_row(f"standalone re-score {metric}",
                                "PASS" if selected and within(selected[metric], found, tol) else "FAIL",
                                found, selected and selected[metric], tol))
    verdict = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS"
    return {"model_id": recipe["model_id"], "verdict": verdict,
            "selected_epoch": selected and selected["epoch"], "checks": checks,
            "not_included": "full10 gate: run the documented commands in REPRODUCE.md"}


# ---------------------------------------------------------------- gameplay launcher

def canonical_module():
    spec = importlib.util.spec_from_file_location("train_qwen_canonical", REPO / "scripts/train_qwen_canonical.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gameplay_paths(recipe: dict) -> tuple[Path, Path]:
    return (input_location("dataset", recipe["inputs"]["dataset"], recipe),
            input_location("qwen_base", recipe["inputs"]["qwen_base"], recipe))


def stage_fetch(recipe: dict, stages: set[str], print_only: bool) -> list[dict]:
    results = []
    for name, spec in needed_inputs(recipe, stages).items():
        commands = fetch_commands(name, spec, recipe)
        identity = check_identity(name, spec, recipe)
        if identity["ok"]:
            if print_only:
                print(f"# {name}: present and verified at {identity['path']}")
            results.append({"input": name, "status": "present", "path": identity["path"]})
            continue
        if print_only:
            print(f"# {name} -> {input_location(name, spec, recipe)}")
            print("\n".join(commands))
            results.append({"input": name, "status": "printed"})
            continue
        for command in commands:
            print(f"+ {command}", flush=True)
            subprocess.run(command, shell=True, check=True)
        identity = check_identity(name, spec, recipe)
        if not identity["ok"]:
            raise RuntimeError(f"{name} fetched but its identity does not match: {identity}")
        results.append({"input": name, "status": "fetched", "path": identity["path"]})
    return results


def stage_check(recipe: dict, stages: set[str], output_dir: Path | None, gpus: list[str]) -> dict:
    overlays = activate_overlays(recipe)
    env = check_env(recipe)
    inputs = [check_identity(name, spec, recipe) for name, spec in needed_inputs(recipe, stages).items()]
    mismatches = recipe_config_mismatches(recipe, load_config(recipe))
    report = {"code": code_check(recipe), "env": env, "overlays": overlays, "inputs": inputs,
              "recipe_config_mismatches": mismatches}
    plan_error = None
    data, snapshot = gameplay_paths(recipe)
    try:
        placeholder = output_dir or Path("/nonexistent/reproduce_check_output")
        report["plan"] = canonical_module().launch(data, snapshot, placeholder, gpus, dry_run=True)
    except Exception as error:  # report every problem at once
        plan_error = f"{type(error).__name__}: {error}"
    report["plan_error"] = plan_error
    report["ok"] = (all(row["ok"] for row in env) and all(row["present"] for row in overlays)
                    and all(row["ok"] for row in inputs) and not mismatches and not plan_error
                    and report["code"]["ok"])
    return report


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stage_train(recipe: dict, recipe_path: Path, output_dir: Path, gpus: list[str], check: dict) -> None:
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; use a new versioned directory")
    data, snapshot = gameplay_paths(recipe)
    record = {"recipe_path": str(recipe_path), "recipe": recipe, "recipe_sha256": sha256(recipe_path),
              "git": git_state(), "versions": {row["package"]: row["found"] for row in check["env"]},
              "overlays": check["overlays"], "inputs": check["inputs"],
              "resolved": {"dataset": str(data), "qwen_snapshot": str(snapshot), "gpus": gpus},
              "argv": sys.argv, "host": socket.gethostname(),
              "started_at": datetime.now(timezone.utc).isoformat()}
    canonical_module().launch(data, snapshot, output_dir, gpus, f"tcp://127.0.0.1:{free_port()}", record=record)


def stage_evaluate(recipe: dict, output_dir: Path, gpus: list[str], rescore: bool, report_path: Path) -> dict:
    rows = read_epoch_metrics(output_dir / recipe["train"]["epoch_metrics"])
    result = None
    if rescore:
        data, _ = gameplay_paths(recipe)
        destination = report_path.parent / "val48_rescore"
        cfg = recipe["evaluate"]["rescore"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0],
                   PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
        subprocess.run([sys.executable, "-m", cfg["module"], "--data", str(data),
                        "--checkpoint", str(output_dir / recipe["train"]["checkpoint_name"]),
                        "--output-dir", str(destination), *cfg["args"]], cwd=REPO, env=env, check=True)
        metrics = json.loads((destination / "metrics.json").read_text())
        result = metrics.get("binary", metrics)
    report = build_report(recipe, rows, result)
    report.update(run_dir=str(output_dir), evaluated_at=datetime.now(timezone.utc).isoformat(),
                  git=git_state())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def stage_export(recipe: dict, output_dir: Path, gate_root: Path, bundle_dir: Path) -> None:
    spec = recipe["export"]
    template = input_location(spec["template_input"], recipe["inputs"][spec["template_input"]], recipe)
    points = spec["operating_points"]
    subprocess.run([sys.executable, str(REPO / spec["script"]), "--run-dir", str(output_dir),
                    "--model-id", f"{output_dir.name}", "--template-bundle", str(template),
                    "--gate-root", str(gate_root), "--output", str(bundle_dir),
                    "--threshold", str(points["threshold"]), "--padding-sec", str(points["padding_sec_per_side"]),
                    "--merge-gap-sec", str(points["merge_gap_sec"])], cwd=REPO, check=True)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path)
    for stage in ("fetch", "check", "train", "evaluate", "export"):
        parser.add_argument(f"--{stage}", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="with --fetch: print the commands, run nothing")
    parser.add_argument("--output-dir", type=Path, help="run directory (new for --train)")
    parser.add_argument("--gpus", help="comma-separated GPU ids (default: 0..hardware.gpus-1)")
    parser.add_argument("--report", type=Path, help="default: <output-dir>/reproduction_report.json")
    parser.add_argument("--skip-rescore", action="store_true", help="--evaluate without the GPU re-score")
    parser.add_argument("--gate-root", type=Path, help="--export: full10 gate outputs (see REPRODUCE.md)")
    parser.add_argument("--bundle-dir", type=Path, help="--export: new bundle directory")
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    stages = {s for s in ("fetch", "check", "train", "evaluate", "export") if getattr(args, s)} or set(DEFAULT_STAGES)
    gpus = (args.gpus or ",".join(map(str, range(recipe["hardware"]["gpus"])))).split(",")
    if stages & {"train", "evaluate", "export"} and not args.output_dir:
        parser.error("--train, --evaluate and --export need --output-dir")
    if "export" in stages and not (args.gate_root and args.bundle_dir):
        parser.error("--export needs --gate-root and --bundle-dir")
    if "fetch" in stages:
        print(json.dumps({"fetch": stage_fetch(recipe, stages, args.print_only)}, indent=2), flush=True)
        if args.print_only:
            return 0
    check = None
    if stages & {"check", "train"}:
        check = stage_check(recipe, stages, args.output_dir if "train" in stages else None, gpus)
        print(json.dumps({"check": check}, indent=2, default=str), flush=True)
        if not check["ok"]:
            print("CHECK FAILED", file=sys.stderr)
            return 2
    if "train" in stages:
        stage_train(recipe, args.recipe, args.output_dir, gpus, check)
    if "evaluate" in stages:
        activate_overlays(recipe)
        report_path = args.report or args.output_dir / "reproduction_report.json"
        report = stage_evaluate(recipe, args.output_dir, gpus, not args.skip_rescore, report_path)
        for row in report["checks"]:
            print(f"{row['status']:8} {row['name']}: expected {row['expected']} observed {row['observed']}")
        print(f"{report['verdict']}  ({report_path})")
        if report["verdict"] != "PASS":
            return 2
    if "export" in stages:
        stage_export(recipe, args.output_dir, args.gate_root, args.bundle_dir)
    return 0


def entry() -> int:
    """Exit 0 PASS/ok, 1 error, 2 FAIL (every stage)."""
    try:
        return main()
    except SystemExit as stop:  # argparse usage errors exit 2; report them as errors (1)
        return 1 if stop.code == 2 else int(stop.code or 0)
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(entry())

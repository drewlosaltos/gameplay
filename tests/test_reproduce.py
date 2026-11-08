"""Recipe loader/validator, recipe-config agreement, and reproduction PASS/FAIL logic (no GPU)."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
RECIPE = REPO / "recipes/v8_val48_qwen672x384_single_stage_fp8_20261006_r3.json"
spec = importlib.util.spec_from_file_location("reproduce", REPO / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


@pytest.fixture
def recipe():
    return reproduce.load_recipe(RECIPE)


def prod_rows(recipe):
    return [dict(row) for row in recipe["expected"]["per_epoch"]]


def test_every_recipe_validates_and_is_named_by_model_id():
    for path in sorted((REPO / "recipes").glob("*.json")):
        recipe = reproduce.load_recipe(path)
        assert path.stem == recipe["model_id"]


def test_prod_recipe_agrees_with_canonical_config(recipe):
    assert reproduce.recipe_config_mismatches(recipe, reproduce.load_config(recipe)) == []


def test_prod_recipe_matches_registry_and_split_contract(recipe):
    yaml = pytest.importorskip("yaml")
    best = yaml.safe_load((REPO / "model_registry.yaml").read_text())["best_model"]
    contract = json.loads((REPO / recipe["select"]["split_contract"]).read_text())
    dataset = recipe["inputs"]["dataset"]
    assert best["model_id"] == recipe["model_id"]
    assert best["checkpoint_sha256"] == recipe["expected"]["prod_reference"]["sha256"]
    assert best["selected_epoch"] == recipe["expected"]["selected_epoch"]
    assert best["f1"] == next(r["f1"] for r in recipe["expected"]["per_epoch"] if r["epoch"] == best["selected_epoch"])
    assert dataset["identity"]["windows_sha256"] == contract["windows_sha256"]
    assert dataset["path"] == contract["dataset_root"]


@pytest.mark.parametrize("change, message", [
    (lambda r: r.pop("expected"), "missing"),
    (lambda r: r.update(schema="skyball_repro_recipe/v0"), "schema"),
    (lambda r: r.update(task="jersey"), "launcher"),
    (lambda r: r["inputs"]["dataset"].update(kind="s3"), "kind"),
    (lambda r: r["inputs"]["qwen_base"].pop("revision"), "revision"),
    (lambda r: r["inputs"]["dataset"].pop("identity"), "identity"),
    (lambda r: r["expected"]["tolerance"].pop("f1_abs"), "tolerance"),
    (lambda r: r["inputs"]["qwen_base"].pop("allow_patterns"), "allow_patterns"),
    (lambda r: r["inputs"]["prod_bundle"].pop("local"), "local"),
    (lambda r: r["inputs"]["prod_bundle"].pop("sha256"), "sha256"),
    (lambda r: r["inputs"]["dataset"]["fetch"].pop("method"), "method"),
    (lambda r: r["env"]["require"].pop("python"), "python"),
    (lambda r: r["expected"].update(selected_epoch=9), "selected_epoch"),
    (lambda r: r["select"].update(rule="best"), "rule"),
    (lambda r: r["code"].pop("min_commit"), "min_commit"),
    (lambda r: r["env"]["pythonpath_overlays"][0].pop("path_env"), "path_env"),
])
def test_validator_rejects_malformed_recipes(recipe, change, message):
    broken = copy.deepcopy(recipe)
    change(broken)
    with pytest.raises(reproduce.RecipeError, match=message):
        reproduce.validate_recipe(broken)


def test_config_drift_is_reported(recipe):
    config = reproduce.load_config(recipe)
    config["args"][config["args"].index("--epochs") + 1] = "5"
    config["dataset"] = "derived/gameplay/other"
    assert set(reproduce.recipe_config_mismatches(recipe, config)) >= {"epochs vs expected.per_epoch", "dataset path"}


def test_prod_metrics_pass(recipe):
    report = reproduce.build_report(recipe, prod_rows(recipe), rescore={"f1": 0.9612})
    assert report["verdict"] == "PASS"
    assert report["selected_epoch"] == 3
    assert all(set(c) == {"name", "status", "observed", "expected", "tolerance"} for c in report["checks"])


def test_report_is_pass_within_tolerance_and_other_epoch_is_info(recipe):
    rows = prod_rows(recipe)
    rows[2]["f1"] -= 0.004
    rows[3]["f1"] += 0.003
    report = reproduce.build_report(recipe, rows)
    assert report["verdict"] == "PASS"
    assert report["selected_epoch"] == 4
    assert {c["status"] for c in report["checks"]} == {"PASS", "INFO", "SKIPPED"}
    assert [c["name"] for c in report["checks"] if c["status"] == "INFO"] == ["selected epoch"]


def test_report_fails_outside_tolerance(recipe):
    rows = prod_rows(recipe)
    rows[0]["f1"] -= 0.006
    failed = [c["name"] for c in reproduce.build_report(recipe, rows)["checks"] if c["status"] == "FAIL"]
    assert failed == ["epoch 1 f1"]


def test_report_fails_on_wrong_population_and_missing_epoch(recipe):
    rows = prod_rows(recipe)
    rows[1]["count"] = 16000
    rows[1]["f1"] = 0.9650  # epoch 2 now best: outside the curve tolerance, selection is INFO
    del rows[3]
    report = reproduce.build_report(recipe, rows)
    failed = {c["name"] for c in report["checks"] if c["status"] == "FAIL"}
    assert report["verdict"] == "FAIL"
    assert failed == {"epoch 2 count", "epoch 2 f1", "epoch 4 f1", "epoch 4 count"}


def test_report_fails_when_rescore_disagrees(recipe):
    report = reproduce.build_report(recipe, prod_rows(recipe), rescore={"f1": 0.95})
    assert report["verdict"] == "FAIL"
    assert [c["name"] for c in report["checks"] if c["status"] == "FAIL"] == ["standalone re-score f1"]


def test_select_epoch_breaks_ties_toward_the_earlier_epoch():
    rows = [{"epoch": 1, "f1": .9}, {"epoch": 2, "f1": .95}, {"epoch": 3, "f1": .95}]
    assert reproduce.select_epoch(rows, {"metric": "f1", "rule": "max_metric"})["epoch"] == 2
    assert reproduce.select_epoch(rows, {"metric": "f1", "rule": "min_metric"})["epoch"] == 1


def test_version_matching():
    assert reproduce.version_matches("2.11.0", "2.11.0+cu130")
    assert reproduce.version_matches("3.12", "3.12.13")
    assert not reproduce.version_matches("0.17.0+cu128", "0.17.0+cu130")
    assert not reproduce.version_matches("2.11.0", "2.11.1")
    assert not reproduce.version_matches("5.6.0", None)


def test_identity_check_reads_the_dataset_manifest(recipe, tmp_path, monkeypatch):
    monkeypatch.setenv("SKYBALL_DATA_ROOT", str(tmp_path))
    spec = copy.deepcopy(recipe["inputs"]["dataset"])
    root = tmp_path / spec["path"] / "metadata"
    root.mkdir(parents=True)
    (root / "windows.jsonl").write_text("{}\n")
    (root / "summary.json").write_text("{}")
    spec["identity"] = {"windows_sha256": reproduce.sha256(root / "windows.jsonl"),
                        "summary_sha256": "0" * 64}
    result = reproduce.check_identity("dataset", spec, recipe)
    assert [c["ok"] for c in result["checks"]] == [True, False]
    assert not result["ok"]


def test_fetch_commands_cover_each_input_kind(recipe, tmp_path, monkeypatch):
    monkeypatch.setenv("SKYBALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("SKYBALL_STAGING_ROOT", raising=False)
    monkeypatch.setenv("SKYBALL_REPRO_CACHE", str(tmp_path / "repro"))
    hf = reproduce.fetch_commands("qwen_base", recipe["inputs"]["qwen_base"], recipe)
    data = reproduce.fetch_commands("dataset", recipe["inputs"]["dataset"], recipe)
    drive = reproduce.fetch_commands("prod_bundle", recipe["inputs"]["prod_bundle"], recipe)
    assert hf[0].startswith("hf download Qwen/Qwen3.6-35B-A3B-FP8 --revision 95a723d0")
    assert "sha256sum -c SHA256SUMS" in data[1] and str(tmp_path / "data/derived/gameplay") in data[3]
    assert str(tmp_path / "unified_staging/derived/gameplay") in data[0]  # staging follows the data root
    assert str(tmp_path / "repro/gameplay/") in drive[0]
    assert drive[0].startswith("rclone copy googdrive:skyball/checkpoints/gameplay/") and "training_checkpoint" in drive[0]
    assert set(reproduce.needed_inputs(recipe, {"check"})) == {"qwen_base", "dataset"}
    assert "prod_bundle" in reproduce.needed_inputs(recipe, {"export"})


def test_train_refuses_an_existing_output_dir(recipe, tmp_path):
    with pytest.raises(FileExistsError):
        reproduce.stage_train(recipe, RECIPE, tmp_path, ["0", "1", "2", "3"], {})


def test_cache_defaults_stay_outside_the_checkout(monkeypatch):
    monkeypatch.delenv("SKYBALL_REPRO_CACHE", raising=False)
    monkeypatch.delenv("SKYBALL_STAGING_ROOT", raising=False)
    monkeypatch.setenv("SKYBALL_DATA_ROOT", "/data/x/unified_dataset")
    assert reproduce.repro_cache() == Path.home() / ".cache/skyball_repro"
    assert reproduce.staging_root() == Path("/data/x/unified_staging")


def test_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce.sys, "argv", ["reproduce.py", str(tmp_path / "missing.json"), "--check"])
    assert reproduce.entry() == 1  # error
    monkeypatch.setattr(reproduce.sys, "argv", ["reproduce.py", str(RECIPE), "--train"])
    assert reproduce.entry() == 1  # usage error (needs --output-dir), not FAIL
    run = tmp_path / "run"
    run.mkdir()
    rows = [dict(r, f1=r["f1"] - 0.01) for r in json.loads(RECIPE.read_text())["expected"]["per_epoch"]]
    (run / "epoch_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(reproduce.sys, "argv", ["reproduce.py", str(RECIPE), "--evaluate", "--skip-rescore",
                                                 "--output-dir", str(run)])
    assert reproduce.entry() == 2  # FAIL
    assert json.loads((run / "reproduction_report.json").read_text())["verdict"] == "FAIL"

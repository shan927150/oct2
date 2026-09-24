"""Shared, standard-library-only contracts for the isolated B/C experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = [42, 43, 44, 45, 46]
PATIENTS = [807, 2085]
ARMS = ["control", "decay"]
DOSES = {"full": 1.0, "dose01": 0.1}
SCHEMA = "pathway2_A_smallpanel_loo_v3"
CONFIG = "smallpanel_config.json"
SCHEDULE = "20@1e-4,20@1e-5"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_name(seed, patient, dose, arm):
    return f"seed{seed}_patient{patient}_{dose}_{arm}"


def expected_cells():
    return [cell_name(s, p, d, a) for s in SEEDS for p in PATIENTS for d in DOSES for a in ARMS]


def disjoint(output, *protected):
    output = Path(output).resolve()
    for p in map(lambda x: Path(x).resolve(), protected):
        if output == p or output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError(f"Output overlaps protected input: {p}")


def verify_hashes(mapping):
    for name, digest in mapping.items():
        if sha(name) != digest:
            raise ValueError(f"Frozen input changed: {name}")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def load_config(root):
    root = Path(root).resolve()
    if sha(root / CONFIG) != (root / 'config.sha256').read_text().strip():
        raise ValueError('Prepared configuration changed')
    cfg = read(root / CONFIG)
    if cfg.get("schema") != "pathway2_A_smallpanel_config_v3" or cfg["root"] != str(root):
        raise ValueError("Not an immutable small-panel v3 attempt")
    if cfg["repo"] != str(REPO) or git("rev-parse", "HEAD") != cfg["git_commit"]:
        raise ValueError("Prepared code/repository changed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked source files are modified")
    if cfg["cells"] != expected_cells() or cfg["resume_epoch"] != 100 or cfg["total_epochs"] != 140 or cfg["decay_schedule"] != SCHEDULE:
        raise ValueError("Prepared experimental contract changed")
    verify_hashes(cfg["input_sha256"])
    return root, cfg


def optimizer_reference(cert, seed, patient, affected=3, epoch=100):
    if cert.get("status") != "complete" or cert.get("extended_epochs") != epoch:
        raise ValueError("Incomplete or wrong-epoch truth replay certificate")
    name = f"shadow_{affected}_seed{seed}_patient{patient}"
    rows = [r for r in cert["checks"] if r.get("role") == "loo" and
            r.get("seed") == seed and r.get("training_instance") == name]
    if len(rows) != 1 or rows[0].get("passed") is not True:
        raise ValueError(f"Missing/duplicate/failed truth optimizer certificate: {name}")
    digest = rows[0].get("continued_optimizer_sha256_at_E_star")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Missing E{epoch} optimizer fingerprint: {name}")
    return digest


def validate_probe_manifest(m, seed, arm):
    expected = dict(schema="pathway2_A_lr_continuation_v1", status="complete", seed=seed,
                    arm=arm, resume_epoch=100, total_epochs=140, extra_epochs=40,
                    base_lr=0.001, weight_decay=1e-5,
                    decay_schedule=SCHEDULE if arm == "decay" else None)
    for k, v in expected.items():
        if m.get(k) != v:
            raise ValueError(f"Probe {seed}/{arm}: {k} does not match {v!r}")
    for k in ("input_files_unchanged", "source_files_unchanged", "dataset_arrays_unchanged"):
        if m.get(k) is not True:
            raise ValueError(f"Probe {seed}/{arm}: {k} failed")
    if m.get("resume_check", {}).get("passed") is not True:
        raise ValueError("Probe resume check failed")


def collect_inputs(cfg):
    """Validate actual source artifacts, then freeze their content hashes."""
    full = Path(cfg["full_dir"])
    e100 = Path(cfg["e100_root"])
    probe = Path(cfg["probe_root"])
    pcfg = read(full / "experiment_config.json")["args"]
    if (pcfg["seeds"] != SEEDS or pcfg["affected_shadow"] != 3 or
            pcfg["shadow_lr"] != 0.001 or pcfg["shadow_epochs"] != 50 or
            pcfg["deletion_mode"] != "fixed_mask" or not pcfg["deterministic"]):
        raise ValueError("Original training config does not match this frozen experiment")
    if read(e100 / "A_STEP2_STATE.json")["E_star"] != 100:
        raise ValueError("E100 truth attempt required")
    panel = {p["patient_id"]: p for p in read(full / "selected_patients.json")["patients"]}
    if [(p, panel[p]["oct_class"]) for p in PATIENTS] != [(807, 1), (2085, 2)]:
        raise ValueError("Frozen patient/class panel changed")
    # The job ledger also records independent A2 damping submissions; it is not
    # a training input. Freeze the endpoint selection and actual truth files.
    paths = [e100 / "frozen_E_star.json"]
    rels = ["experiment_config.json", "selected_patients.json", "splits/fresh_patient_split.json"]
    paths += [full / n for n in rels]
    for dose in DOSES:
        d = e100 / f"shadow3_{dose}"
        a = read(d / "experiment_config.json")["args"]
        for k in ("seeds", "affected_shadow", "shadow_lr", "shadow_batch_size", "deterministic", "deletion_mode"):
            if a[k] != pcfg[k]:
                raise ValueError(f"E100 {dose}: {k} differs from original config")
        if a["shadow_epochs"] != 100 or a["deletion_weight"] != DOSES[dose]:
            raise ValueError("Wrong E100 truth horizon/deletion dose")
        if read(d / rels[2]) != read(full / rels[2]):
            raise ValueError(f"E100 {dose}: split mismatch")
        original_panel, truth_panel = read(full / rels[1]), read(d / rels[1])
        for k in ('patients', 'split_sha256'):
            if truth_panel[k] != original_panel[k]:
                raise ValueError(f"E100 {dose}: panel {k} mismatch")
        certpath = d / "route_a_epoch50_replay_checks.json"
        cert = read(certpath)
        if read(d / "experiment_summary.json").get("status") != "complete":
            raise ValueError("Truth not complete")
        paths += [d / n for n in rels] + [certpath, d / "experiment_summary.json"]
        for seed in SEEDS:
            paths += [d / "checkpoints" / f"shadow_3_baseline_seed{seed}.pt",
                      d / f"stage1_order_seed{seed}.npz"]
            for pid in PATIENTS:
                optimizer_reference(cert, seed, pid)
                paths.append(d / "checkpoints" / f"shadow_3_seed{seed}_patient{pid}.pt")
    datasets, facts = [], {}
    for seed in SEEDS:
        for arm in ARMS:
            d = probe / f"shadow3_seed{seed}_{arm}"
            m = read(d / "manifest.json")
            validate_probe_manifest(m, seed, arm)
            inputs = read(d / "input_sha256.json")
            verify_hashes(inputs)
            for n in ("experiment_config.json", "splits/fresh_patient_split.json"):
                if inputs.get(str(full / n)) != sha(full / n):
                    raise ValueError("Probe baseline belongs to different original inputs")
            for n in ("05_end_to_end_patient_loo_pilot.py", "07_cross_stage_score_ladder.py",
                      "11_stage1_diagnostics.py", "stage1_diagnostic_core.py", "20_A_convergence_curve.py"):
                rel = "scripts/cross_stage/" + n
                if m["source_sha256"].get(rel) != sha(REPO / rel):
                    raise ValueError(f"Training implementation differs from probe: {rel}")
            datasets.append(read(d / "dataset_fingerprint.json"))
            facts[f"{seed}/{arm}"] = {k: m[k] for k in
                ("epoch_order_sha256", "final_rng_sha256", "resume_check", "git_commit")}
            paths += [d / n for n in ("manifest.json", "input_sha256.json", "dataset_fingerprint.json",
                                     "curve.csv", "checkpoints/epoch140.pt")]
            paths += list(map(Path, inputs))
    if not all(d == datasets[0] for d in datasets):
        raise ValueError("Probe dataset fingerprints differ")
    for seed in SEEDS:
        c, d = facts[f"{seed}/control"], facts[f"{seed}/decay"]
        for key in ("epoch_order_sha256", "final_rng_sha256", "resume_check"):
            if c[key] != d[key]:
                raise ValueError("Probe arms do not share the same training/RNG stream")
    return {str(p): sha(p) for p in sorted(set(paths))}, datasets[0], facts


def check_baseline_metadata(payload, seed, arm, facts):
    md = payload["metadata"]
    expected = dict(seed=seed, arm=arm, epoch=140, resume_epoch=100, total_epochs=140,
                    base_lr=0.001, decay_schedule=SCHEDULE if arm == "decay" else None,
                    source="pathway2_A_lr_continuation_v1",
                    epoch_order_sha256=facts["epoch_order_sha256"],
                    rng_fingerprint=facts["final_rng_sha256"])
    for k, v in expected.items():
        if md.get(k) != v:
            raise ValueError(f"Baseline metadata mismatch {seed}/{arm}: {k}")
    if not md.get("optimizer_state") or not md.get("rng_states"):
        raise ValueError("Baseline checkpoint lacks continuation state")


def validate_cell_manifest(m, cfg, name):
    if m.get("schema") != SCHEMA or m.get("cell") != name:
        raise ValueError(f"Wrong cell schema/name: {name}")
    if m.get("all_cells") != expected_cells():
        raise ValueError("Cell panel differs from frozen 40-cell panel")
    for k in ("resume_epoch", "total_epochs", "git_commit"):
        if m.get(k) != cfg[k]:
            raise ValueError(f"Cell {name}: {k} mismatch")
    if m.get("config_sha256") != cfg["config_sha256"]:
        raise ValueError("Cell was run against another prepared configuration")
    for k in ("replay_e100_passed", "input_files_unchanged", "source_files_unchanged", "dataset_arrays_unchanged"):
        if m.get(k) is not True:
            raise ValueError(f"Cell {name}: {k} did not pass")
    if m.get("status") != "complete":
        raise ValueError(f"Cell {name} incomplete")
    expected = cell_name(m['seed'], m['patient_id'], m['dose'], m['arm'])
    if name != expected or m.get('panel') != PATIENTS or m.get('affected_shadow') != 3:
        raise ValueError('Cell coordinates/panel mismatch')
    for key, value in dict(base_lr=.001, weight_decay=1e-5, deletion_weight=DOSES[m['dose']],
                           decay_schedule=SCHEDULE if m['arm'] == 'decay' else None).items():
        if m.get(key) != value:
            raise ValueError(f'Cell recipe mismatch: {key}')

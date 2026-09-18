#!/usr/bin/env python3
"""Accept execution/integrity separately from scientific B1 derivative resolution."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import subprocess
import tarfile


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def accounting(text, expected):
    states = {}
    for line in text.splitlines():
        parts = [v.strip() for v in line.split("|")]
        if len(parts) >= 3:
            states[parts[0]] = parts[1:3]
    for job in expected:
        require(states.get(job) == ["COMPLETED", "0:0"], f"sacct not COMPLETED/0:0: {job}: {states.get(job)}")
    return {j: states[j] for j in expected}


def validate_result(folder, code, seed):
    m = read(folder/"manifest.json")
    require(m["schema"] == "pathway2_b1_smoke_v2", "Wrong B1 schema")
    require(m["status"] == "complete" and m["input_files_unchanged"], "Execution/input integrity failed")
    require(m["git_commit"] == code and m["args"]["seed"] == seed, "Wrong commit or seed")
    require(m["tf32"] == m["tf32_after"], "TF32 settings changed during B1")
    require(m["args"]["patients"] == [807,2085,1369], "Production patient coverage differs")
    require(m["args"]["prefix_epochs"] == [0,1,5], "Production prefix coverage differs")
    require(m["args"]["exponents"] == [10,14,18,22,24], "Production float32 ladder differs")
    require(m["args"]["exponents64"] == [10,14,18,22,24,28,32], "Production float64 ladder differs")
    for q, digest in read(folder/"input_sha256.json").items():
        p = Path(q)
        require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest,
                f"Frozen input changed since run: {p}")
    summary = read(folder/"b1_summary.json")
    gates = summary["gates"]
    g1 = gates["G1"]
    require(g1["status"] == "PASS_FULL_REPLAY" and g1["final_parameters_bitwise_equal"], "G1 failed")
    refs = gates["G1"]["checked_against_original_epoch_checkpoints"]
    require(refs and refs[-1]["epoch"] == g1["original_epochs"], "G1 final epoch missing")
    require(all(r[k] for r in refs for k in ("parameters_bitwise_equal","optimizer_bitwise_equal","rng_bitwise_equal")),
            "G1 model/Adam/RNG mismatch")
    payload = read(folder/"b1_rows.json")
    rows, prefixes = payload["rows"], payload["prefixes"]
    per_epoch = payload["batches_per_epoch"]
    require({1,per_epoch,5*per_epoch}.issubset(prefixes), "Missing requested prefixes")
    expected = {(seed,p,k,lane,e) for p in (807,2085,1369) for k in prefixes
                for lane in ("functional32","functional64","native32")
                for e in m["args"]["exponents64" if lane == "functional64" else "exponents"]}
    actual = {(r["seed"],r["patient_id"],r["prefix_steps"],r["lane"],r["exponent"]) for r in rows}
    require(actual == expected and len(rows) == len(expected) == m["n_rows"], "Incomplete/duplicate B1 rows")
    expected_g3 = {(lane,k) for lane in ("functional32","functional64") for k in prefixes}
    require({(r["lane"],r["prefix_steps"]) for r in gates["G3"]} == expected_g3, "G3 coverage missing")
    require(all(r["all_state_tangents_exactly_zero"] and r["primal_bitwise_equal"] for r in gates["G3"]), "G3 failed")
    expected_g4 = {(p,k,lane) for p in (807,2085,1369) for k in prefixes
                   for lane in ("functional32","functional64","native32")}
    require({(r["patient_id"],r["prefix_steps"],r["lane"]) for r in gates["G4"]} == expected_g4, "G4 coverage missing")
    require(all(r["status"] in ("ZERO_CONTROL","RESOLVED_SAMPLED_BAND","UNRESOLVED_AT_TESTED_DOSES")
                for r in gates["G4"]), "Invalid scientific verdict")
    require(summary["ready_for_B2"] is False, "Short-prefix run cannot certify B2")
    return m, {"original_numerical_fidelity": summary["original_numerical_trajectory_matched_at_sampled_states"],
               "prefix_derivative_verdicts": gates["G4"], "ready_for_B2": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    submission = args.submission.expanduser().resolve(strict=True)
    root = submission.parent
    ids = dict(line.split("=",1) for line in submission.read_text().splitlines() if "=" in line)
    require(set(ids) == {"GPU_TEST","B1_ARRAY","CODE"}, "Invalid submission fields")
    require(all(ids[k].isdigit() for k in ("GPU_TEST","B1_ARRAY")) and re.fullmatch(r"[0-9a-f]{40}",ids["CODE"]),
            "Invalid IDs/commit")
    result = subprocess.run(["sacct","--array","-X","-n","-P","-j",ids["GPU_TEST"]+","+ids["B1_ARRAY"],
                             "--format=JobID%40,State%30,ExitCode"], check=True, capture_output=True,text=True)
    states = accounting(result.stdout,[ids["GPU_TEST"],ids["B1_ARRAY"]+"_0",ids["B1_ARRAY"]+"_1"])
    checks = [(ids["GPU_TEST"],"single"),(ids["B1_ARRAY"],"0"),(ids["B1_ARRAY"],"1")]
    files = [submission]
    for job,task in checks:
        f = root/f"postcheck_{job}_{task}.json"
        post = read(f)
        require(post["status"] == "PASS_FILE_INTEGRITY_ONLY" and post["commit"] == ids["CODE"], "Postcheck failed")
        files.append(f)
    verdicts, data = {}, None
    for seed in (42,43):
        folder = root/f"b1_{ids['B1_ARRAY']}_seed{seed}"
        manifest, verdicts[str(seed)] = validate_result(folder,ids["CODE"],seed)
        if data is not None:
            require(data == manifest["loaded_data_sha256"], "Loaded data arrays differ across seeds")
        data = manifest["loaded_data_sha256"]
        for f in folder.rglob("*"):
            require(not f.is_symlink(), "Unexpected result symlink")
            if f.is_file() and f.suffix in (".json",".csv"):
                files.append(f)
    repo = Path(__file__).resolve().parents[2]
    logs = [repo/"logs"/f"oct_b1_test-{ids['GPU_TEST']}.{ext}" for ext in ("out","err")]
    logs += [repo/"logs"/f"oct_b1-{ids['B1_ARRAY']}_{task}.{ext}" for task in (0,1) for ext in ("out","err")]
    require(all(p.is_file() and not p.is_symlink() for p in logs), "Job logs missing")
    require("B1_GPU_TESTS_PASSED" in logs[0].read_text(), "GPU test success marker missing")
    report = {"execution_and_integrity":"PASS","jobs":ids,"sacct":states,"scientific_verdicts":verdicts,
              "note":"UNRESOLVED and approximate functional/native fidelity are recorded findings, not acceptance of an original-algorithm derivative. No B2 or alpha=1 claim."}
    output = args.output.expanduser().resolve()
    require(not output.exists() and not output.is_relative_to(root), "Use a new archive outside the run root")
    with tarfile.open(output,"x:gz") as archive:
        for f in sorted(set(files)):
            archive.add(f,arcname=str(f.relative_to(root)))
        for f in logs:
            archive.add(f,arcname="logs/"+f.name)
        for name,blob in (("acceptance.json",json.dumps(report,indent=2).encode()),("sacct.txt",result.stdout.encode())):
            entry=tarfile.TarInfo(name); entry.size=len(blob)
            archive.addfile(entry,io.BytesIO(blob))
    print(json.dumps({**report,"archive":str(output)},indent=2))


if __name__ == "__main__":
    main()

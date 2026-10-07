"""Local receipts plus Slurm state must agree before a retry is submitted."""
import json
from pathlib import Path
import subprocess

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def receipts(root):
    root = Path(root)
    if (root / 'submission_pending.json').exists():
        raise RuntimeError('Unknown sbatch outcome: reconcile submission_pending.json with squeue/sacct before retrying; do not delete blindly')
    path = root / 'submission.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def require_terminal(records):
    if not records:
        return
    ids = sorted({str(r['job_id']) for r in records})
    live = subprocess.check_output(['squeue', '-r', '-h', '-j', ','.join(ids), '-o', '%i'], text=True).strip()
    if live:
        raise RuntimeError('Prior jobs are still queued/running; retry withheld: ' + live[:300])
    raw = subprocess.check_output(['sacct', '-n', '-X', '-P', '-j', ','.join(ids),
                                   '--format=JobIDRaw,State'], text=True)
    seen = set()
    for line in raw.splitlines():
        parts = line.strip().split('|')
        if len(parts) < 2 or '.' in parts[0]:
            continue
        job, state = parts[0], parts[1].split()[0].rstrip('+')
        parent = job.split('_')[0]
        if parent in ids:
            seen.add(parent)
            if state not in TERMINAL:
                raise RuntimeError(f'Accounting not terminal: {job} {state}')
    if set(ids) != seen:
        raise RuntimeError('Accounting is missing prior jobs; wait for sacct before retrying')


def active_ids(records):
    ids = sorted({str(r['job_id']) for r in records})
    if not ids:
        return []
    raw = subprocess.check_output(['squeue', '-r', '-h', '-j', ','.join(ids), '-o', '%i'], text=True)
    return sorted({line.strip().split('_')[0] for line in raw.splitlines() if line.strip()} & set(ids))

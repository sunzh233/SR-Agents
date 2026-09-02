"""Parallel per-instance inference with append-based resume.

Orchestrates a Provider × Engine pair over a list of instances, writing
one :class:`~sragents.infer.schema.InferenceRecord` per line to the
output JSONL. Re-invocation with the same ``--output`` path skips
already-completed instances automatically.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from sragents.config import model_short_name
from sragents.infer.base import InferenceEngine, SkillProvider
from sragents.infer.schema import InferenceRecord

_write_lock = threading.Lock()


def _already_done(
    out_path: Path, retry_ids: set[str] | None = None,
) -> set[str]:
    """Return successful instance IDs and make selected failed rows retryable.

    A provider/model exception is a record of an attempt, not completion.
    Resume therefore atomically removes failed rows belonging to this call
    (and any trailing partial line), while retaining rows from other calls
    that share the output file.  The latter matters when adapter waves append
    to one result file: a later wave cannot retry an earlier wave after its
    adapters have been unloaded.
    """
    if not out_path.exists():
        return set()
    done: set[str] = set()
    seen: set[str] = set()
    with open(out_path, "rb") as f:
        data = f.read()
    kept: list[bytes] = []
    for raw in data.splitlines(keepends=True):
        stripped = raw.decode("utf-8", errors="replace").strip()
        if not stripped:
            continue
        if not raw.endswith(b"\n"):
            # Trailing partial line — don't count, don't advance.
            break
        try:
            rec = json.loads(stripped)
            instance_id = rec["instance_id"]
            model_failed = bool(rec.get("meta", {}).get("failed"))
            if instance_id in seen:
                continue
            failed = bool(
                rec.get("error")
                or (not str(rec.get("raw_output", "")).strip()
                    and not model_failed)
            )
            if failed and (retry_ids is None or instance_id in retry_ids):
                continue
            seen.add(instance_id)
            if failed:
                kept.append((stripped + "\n").encode("utf-8"))
                continue
            done.add(instance_id)
            kept.append((stripped + "\n").encode("utf-8"))
        except (json.JSONDecodeError, KeyError):
            # Stop at the first malformed line; later bytes cannot be trusted.
            break
    cleaned = b"".join(kept)
    if cleaned != data:
        temporary = out_path.with_suffix(out_path.suffix + ".resume.tmp")
        temporary.write_bytes(cleaned)
        temporary.replace(out_path)
    return done


def _append(fout, record: InferenceRecord) -> None:
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with _write_lock:
        fout.write(line)
        fout.flush()


def run_many(
    instances: list[dict],
    provider: SkillProvider,
    engine: InferenceEngine,
    client,
    model: str,
    output_path: Path,
    label: str,
    workers: int = 32,
    engine_kwargs: dict | None = None,
) -> None:
    """Run provider×engine on ``instances``, streaming results to ``output_path``.

    Skips instances already present in ``output_path`` (per-instance resume).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = _already_done(
        output_path, {str(instance["instance_id"]) for instance in instances},
    )
    pending = [i for i in instances if i["instance_id"] not in done]

    if not pending:
        print(f"  all {len(instances)} instances already complete → {output_path}")
        return
    if done:
        print(f"  resuming: {len(done)}/{len(instances)} done, {len(pending)} remaining")

    model_name = model_short_name(model)
    engine_kwargs = engine_kwargs or {}

    def _one(inst: dict) -> InferenceRecord:
        try:
            skills = provider.provide(inst)
            result = engine.run(inst, skills, client, model, **engine_kwargs)
            if not str(result.raw_output).strip() and not result.meta.get("failed"):
                raise RuntimeError("model returned empty output")
            return InferenceRecord(
                instance_id=inst["instance_id"],
                dataset=inst["dataset"],
                method=label,
                model=model_name,
                raw_output=result.raw_output,
                transcript=result.transcript,
                skill_ids_used=result.skill_ids_used,
                meta=result.meta,
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n  ERROR on {inst['instance_id']}: {e}", file=sys.stderr)
            return InferenceRecord(
                instance_id=inst["instance_id"],
                dataset=inst["dataset"],
                method=label,
                model=model_name,
                raw_output="",
                error=str(e),
            )

    n_workers = max(workers, 1)
    with open(output_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_one, i): i for i in pending}
            with tqdm(total=len(pending), desc=f"  {output_path.name}") as bar:
                for fut in as_completed(futures):
                    _append(fout, fut.result())
                    bar.update(1)

    print(f"  wrote {len(pending)} records → {output_path}")

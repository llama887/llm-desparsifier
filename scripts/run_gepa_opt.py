#!/usr/bin/env python3
"""Run GEPA optimization on the latest dataset and update the active prompt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import dspy

from llm_desparsifier.rewards import RewardSynthesizer, configure_portkey_lm
from llm_desparsifier.rewards.parser import CONSTRAINTS_TEXT
from llm_desparsifier.utils import get_active_prompt_path, write_active_prompt

STATE_DEFAULT = Path("artifacts/gepa_state")
BASE_PROMPT_PATH = Path("llm_desparsifier/rewards/prompts/base_reward_prompt.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA optimization and refresh the active prompt")
    parser.add_argument("--state-root", type=Path, default=STATE_DEFAULT, help="Shared GEPA state directory")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional override JSONL dataset path")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light", help="GEPA auto budget")
    parser.add_argument("--max-full-evals", type=int, default=None, help="Optional GEPA max_full_evals override")
    parser.add_argument("--num-threads", type=int, default=8, help="Threads for GEPA evaluation")
    return parser.parse_args()


def find_pending_iteration(state_root: Path) -> Optional[Path]:
    candidates = []
    for child in state_root.glob("iter-*"):
        if not child.is_dir():
            continue
        if not (child / "ready_for_gepa").exists():
            continue
        if (child / "prompt_ready").exists():
            continue
        if not (child / "train_dense.jsonl").exists():
            continue
        candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.name)
    return candidates[-1]


def load_prompt_payload(state_root: Path) -> tuple[str, Optional[Dict[str, Any]]]:
    prompt_path = get_active_prompt_path(state_root)
    if prompt_path.exists():
        payload = json.loads(prompt_path.read_text())
        text = payload.get("constraints_text")
        if isinstance(text, str) and text.strip():
            return text, payload.get("synthesizer_state")
    if BASE_PROMPT_PATH.exists():
        return BASE_PROMPT_PATH.read_text(), None
    return CONSTRAINTS_TEXT, None


def select_dataset(args: argparse.Namespace, state_root: Path) -> tuple[Path, Optional[Path]]:
    if args.dataset is not None:
        dataset_path = args.dataset.expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        iteration_dir = dataset_path.parent
        return dataset_path, iteration_dir
    iteration_dir = find_pending_iteration(state_root)
    if iteration_dir is None:
        raise RuntimeError("No iteration directories with ready_for_gepa found.")
    dataset_path = iteration_dir / "train_dense.jsonl"
    return dataset_path, iteration_dir


def load_dataset_records(dataset_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Dataset {dataset_path} is empty")
    return records


def build_examples(records: Sequence[Mapping[str, Any]], constraints_text: str) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    for record in records:
        env_description = record.get("env_description") or record.get("env_id") or "Unknown env"
        example = dspy.Example(
            env_description=env_description,
            constraints=constraints_text,
        ).with_inputs("env_description", "constraints")
        example.sparse_return_curve = record.get("sparse_return_curve", [])
        example.reflection_text = record.get("reflection_text", "")
        example.metadata = record
        examples.append(example)
    return examples


def split_examples(examples: List[dspy.Example]) -> tuple[List[dspy.Example], List[dspy.Example]]:
    if len(examples) <= 1:
        return examples, examples
    valset = examples[::2]
    trainset = examples[1::2]
    if not trainset:
        trainset = valset
    if not valset:
        valset = trainset
    return trainset, valset


def metric_with_feedback(example: dspy.Example, prediction: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    curve = getattr(example, "sparse_return_curve", []) or []
    final_sparse = float(curve[-1]) if curve else 0.0
    feedback = getattr(example, "reflection_text", "")
    return dspy.Prediction(score=final_sparse, feedback=feedback)


class RewardPromptProgram(dspy.Module):
    def __init__(self, constraints_text: str, synthesizer_state: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self.constraints_text = constraints_text
        self.synthesizer = RewardSynthesizer()
        if synthesizer_state:
            self.synthesizer.gen.load_state(synthesizer_state)

    def forward(self, env_description: str, constraints: Optional[str] = None):
        text = constraints or self.constraints_text
        reward_code = self.synthesizer(env_description=env_description, constraints=text)
        return dspy.Prediction(reward_code=reward_code)


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_iteration_outputs(
    iteration_dir: Optional[Path],
    prompt_payload: Mapping[str, Any],
    report_payload: Mapping[str, Any],
) -> None:
    if iteration_dir is None:
        return
    optimized_path = iteration_dir / "optimized_prompt.json"
    with optimized_path.open("w", encoding="utf-8") as handle:
        json.dump({"prompt": prompt_payload, "report": report_payload}, handle, indent=2, sort_keys=True)
    (iteration_dir / "prompt_ready").touch()


def run():
    args = parse_args()
    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)

    dataset_path, iteration_dir = select_dataset(args, state_root)
    records = load_dataset_records(dataset_path)

    constraints_text, synthesizer_state = load_prompt_payload(state_root)
    examples = build_examples(records, constraints_text)
    trainset, valset = split_examples(examples)

    base_lm = configure_portkey_lm()
    dspy.configure(lm=base_lm)

    program = RewardPromptProgram(constraints_text, synthesizer_state)

    compiler = dspy.GEPA(
        metric=metric_with_feedback,
        auto=args.auto,
        max_full_evals=args.max_full_evals,
        num_threads=args.num_threads,
        reflection_lm=base_lm,
        track_stats=True,
    )

    optimized_program = compiler.compile(program, trainset=trainset, valset=valset)

    prompt_payload = {
        "constraints_text": constraints_text,
        "synthesizer_state": optimized_program.synthesizer.gen.dump_state(),
        "updated_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "source_iteration": None if iteration_dir is None else iteration_dir.name,
    }
    write_active_prompt(state_root, prompt_payload)

    report_payload = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": compute_file_hash(dataset_path),
        "train_examples": len(trainset),
        "val_examples": len(valset),
        "auto": args.auto,
        "max_full_evals": args.max_full_evals,
        "num_threads": args.num_threads,
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    write_iteration_outputs(iteration_dir, prompt_payload, report_payload)

    print("[run_gepa_opt] Updated active prompt at", get_active_prompt_path(state_root))


if __name__ == "__main__":
    run()

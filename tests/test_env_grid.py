from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_GRID_PATH = REPO_ROOT / "configs" / "gepa_envs.yaml"


def _load_env_grid():
    data = yaml.safe_load(ENV_GRID_PATH.read_text())
    jobs = data.get("jobs", [])
    eval_jobs = data.get("eval_jobs", [])
    return jobs, eval_jobs


def _env_ids(entries):
    return {entry["env_id"] for entry in entries}


def test_gepa_env_grid_training_and_eval_split():
    jobs, eval_jobs = _load_env_grid()

    train_envs = _env_ids(jobs)
    eval_envs = _env_ids(eval_jobs)

    assert train_envs, "Expected training jobs in gepa_envs.yaml"
    assert eval_envs, "Expected eval_jobs section in gepa_envs.yaml"
    assert train_envs.isdisjoint(eval_envs), "Training and eval envs must be disjoint"

    expected_train = {
        "XLand-MiniGrid-R1-11x11",
        "XLand-MiniGrid-R1-13x13",
        "XLand-MiniGrid-R1-15x15",
        "XLand-MiniGrid-R2-11x11",
        "XLand-MiniGrid-R2-13x13",
        "XLand-MiniGrid-R2-15x15",
        "XLand-MiniGrid-R4-11x11",
        "XLand-MiniGrid-R4-13x13",
        "XLand-MiniGrid-R4-15x15",
        "XLand-MiniGrid-R6-17x17",
        "XLand-MiniGrid-R9-19x19",
    }
    expected_eval = {
        "XLand-MiniGrid-R1-9x9",
        "XLand-MiniGrid-R1-17x17",
        "XLand-MiniGrid-R2-9x9",
        "XLand-MiniGrid-R2-17x17",
        "XLand-MiniGrid-R4-9x9",
        "XLand-MiniGrid-R4-17x17",
        "XLand-MiniGrid-R6-13x13",
        "XLand-MiniGrid-R6-19x19",
        "XLand-MiniGrid-R9-16x16",
        "XLand-MiniGrid-R9-25x25",
    }

    assert train_envs == expected_train
    assert eval_envs == expected_eval

    for env_id in train_envs | eval_envs:
        assert env_id.startswith("XLand-MiniGrid-")

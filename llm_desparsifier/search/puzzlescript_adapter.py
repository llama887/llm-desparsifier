"""PuzzleScript state adapter for GEPA heuristic synthesis.

Mirrors xland_adapter.py: extracts full-information context from the
C++ PuzzleScript engine so that LLM-generated heuristic functions can
reason about object positions, win conditions, and game rules.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def decode_object_grid(
    engine,
    id_dict: list[str],
) -> dict[str, list[tuple[int, int]]]:
    """Decode the engine's bitpacked state into {obj_name: [(x, y), ...]}."""
    w, h = engine.get_width(), engine.get_height()
    n_objs = engine.get_object_count()
    stride = (n_objs + 31) // 32
    objs = np.array(engine.get_objects(), dtype=np.int32)

    grid: dict[str, list[tuple[int, int]]] = {}
    for obj_idx, obj_name in enumerate(id_dict):
        word = obj_idx // 32
        bit = obj_idx % 32
        positions = []
        for x in range(w):
            for y in range(h):
                flat_idx = (x * h + y) * stride
                if flat_idx + word < len(objs) and (
                    int(objs[flat_idx + word]) & (1 << bit)
                ):
                    positions.append((x, y))
        if positions:
            grid[obj_name] = positions
    return grid


def render_ascii(
    object_grid: dict[str, list[tuple[int, int]]],
    width: int,
    height: int,
) -> str:
    """Render a compact ASCII representation of the game state."""
    CHAR_MAP = {
        "wall": "#",
        "player": "@",
        "crate": "$",
        "target": ".",
        "box": "$",
        "goal": ".",
        "floor": " ",
        "background": " ",
    }
    # Build priority: non-background objects override background
    cell = {}
    for name, positions in object_grid.items():
        ch = CHAR_MAP.get(name.lower(), name[0].upper())
        priority = 0 if name.lower() == "background" else 1
        for pos in positions:
            key = (pos[0], pos[1])
            if key not in cell or priority > cell[key][1]:
                cell[key] = (ch, priority)

    rows = []
    for y in range(height):
        row = ""
        for x in range(width):
            c, _ = cell.get((x, y), (" ", -1))
            row += c
        rows.append(row)
    return "\n".join(rows)


def build_puzzlescript_ctx(
    engine,
    compiled_json: dict[str, Any],
) -> dict[str, Any]:
    """Build the full-information context dict for an LLM heuristic.

    This is the PuzzleScript analogue of xland_adapter.build_heuristic_ctx().
    The heuristic function receives this as `ctx`.

    Returns a dict with:
      - game_title: str
      - grid_width, grid_height: int
      - object_names: list[str]
      - object_positions: {name: [(x,y), ...]}
      - win_conditions: list[dict]
      - win_conditions_text: str (human-readable)
      - ascii_state: str
      - score: float (built-in heuristic value, lower=closer to win)
      - score_normalized: float (0-1, higher=closer to win)
      - is_winning: bool
      - action_names: {0: "up", 1: "left", ...}
      - n_rules: int
    """
    id_dict = engine.get_id_dict()
    w, h = engine.get_width(), engine.get_height()
    object_grid = decode_object_grid(engine, id_dict)

    # Parse win conditions from compiled JSON
    raw_wcs = compiled_json.get("winconditions", [])
    wc_texts = []
    for wc in raw_wcs:
        mask1_objs = []
        mask2_objs = []
        for idx, name in enumerate(id_dict):
            bit = 1 << idx
            if any(bit & m for m in wc.get("mask1", [])):
                mask1_objs.append(name)
            if any(bit & m for m in wc.get("mask2", [])):
                mask2_objs.append(name)
        num = wc.get("num", 1)
        if num == -1:
            desc = f"no {', '.join(mask1_objs)} on {', '.join(mask2_objs)}"
        elif num == 0:
            desc = f"some {', '.join(mask1_objs)} on {', '.join(mask2_objs)}"
        else:
            desc = f"all {', '.join(mask1_objs)} on {', '.join(mask2_objs)}"
        wc_texts.append(desc)

    ascii_state = render_ascii(object_grid, w, h)

    return {
        "game_title": compiled_json.get("title", "unknown"),
        "grid_width": w,
        "grid_height": h,
        "object_names": id_dict,
        "object_positions": object_grid,
        "win_conditions": raw_wcs,
        "win_conditions_text": "; ".join(wc_texts) if wc_texts else "unknown",
        "ascii_state": ascii_state,
        "score": engine.get_score(),
        "score_normalized": engine.get_score_normalized(),
        "is_winning": engine.is_winning(),
        "action_names": {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"},
        "n_rules": len(compiled_json.get("rules", [])),
    }


def extract_rules_text(game_text: str) -> str:
    """Extract the RULES section from raw PuzzleScript source.

    PuzzleScript rules are the core mechanic — they define how objects
    interact (pushing, swapping, gravity, collapse, teleportation, etc.).
    """
    lines = []
    in_rules = False
    stop_sections = {
        "WINCONDITIONS", "LEVELS", "SOUNDS", "COLLISIONLAYERS", "LEGEND",
    }
    for line in game_text.split("\n"):
        stripped = line.strip().upper()
        if stripped == "RULES":
            in_rules = True
            continue
        if in_rules and stripped in stop_sections:
            break
        if in_rules:
            clean = line.strip()
            # Skip comments and decoration
            if clean and not clean.startswith("(") and clean != "======" \
                    and not all(c == "=" for c in clean):
                lines.append(clean)
    return "\n".join(lines)


def build_env_description(
    compiled_json: dict[str, Any],
    id_dict: list[str],
    game_text: str = "",
) -> str:
    """Build a human-readable environment description for the LLM prompt.

    Includes the game's unique PuzzleScript rules so the LLM can reason
    about mechanics (gravity, swapping, collapsing, etc.) rather than
    defaulting to generic distance heuristics.
    """
    title = compiled_json.get("title", "Unknown PuzzleScript Game")
    obj_names = [name for name in id_dict if name.lower() != "background"]

    raw_wcs = compiled_json.get("winconditions", [])
    wc_texts = []
    for wc in raw_wcs:
        mask1_objs = []
        mask2_objs = []
        for idx, name in enumerate(id_dict):
            bit = 1 << idx
            if any(bit & m for m in wc.get("mask1", [])):
                mask1_objs.append(name)
            if any(bit & m for m in wc.get("mask2", [])):
                mask2_objs.append(name)
        num = wc.get("num", 1)
        if num == -1:
            desc = f"No {', '.join(mask1_objs)} should be on {', '.join(mask2_objs)}"
        elif num == 0:
            desc = f"At least one {', '.join(mask1_objs)} must be on {', '.join(mask2_objs)}"
        else:
            desc = f"All {', '.join(mask1_objs)} must be on {', '.join(mask2_objs)}"
        wc_texts.append(desc)

    rules_text = extract_rules_text(game_text) if game_text else "(rules not available)"

    lines = [
        f"Game: {title}",
        f"Objects: {', '.join(obj_names)}",
        f"Actions: up(0), left(1), down(2), right(3), action(4)",
        f"Win conditions: {'; '.join(wc_texts)}",
        "",
        "PuzzleScript Rules (these define the unique mechanics of this game):",
        rules_text,
        "",
        "The ctx dict contains:",
        "  ctx['object_positions']: dict mapping object names to lists of (x,y) positions",
        "  ctx['grid_width'], ctx['grid_height']: grid dimensions",
        "  ctx['win_conditions_text']: human-readable win conditions",
        "  ctx['ascii_state']: text rendering of current state",
        "  ctx['is_winning']: whether the current state is a win",
        "  ctx['object_names']: list of all object type names",
    ]
    return "\n".join(lines)

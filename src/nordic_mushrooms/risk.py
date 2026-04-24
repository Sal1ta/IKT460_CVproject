from __future__ import annotations

import csv
from pathlib import Path

RISK_ORDER = ["edible", "conditionally_edible", "poisonous", "deadly", "unknown"]
SAFE_RISKS = {"edible", "conditionally_edible"}
UNSAFE_RISKS = {"poisonous", "deadly"}


def normalize_species_key(value: str) -> str:
    cleaned = value.strip().lower().replace("-", " ").replace("_", " ")
    normalized = "_".join(part for part in cleaned.split() if part)
    return normalized


def normalize_risk_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "edible": "edible",
        "safe": "edible",
        "conditionally_edible": "conditionally_edible",
        "conditional_edible": "conditionally_edible",
        "needs_cooking": "conditionally_edible",
        "poisonous": "poisonous",
        "toxic": "poisonous",
        "inedible": "poisonous",
        "deadly": "deadly",
        "deadly_poisonous": "deadly",
        "lethal": "deadly",
        "unknown": "unknown",
    }
    return aliases.get(normalized, "unknown")


def load_risk_map(path: str | Path) -> dict[str, str]:
    risk_path = Path(path)
    if not risk_path.exists():
        return {}

    mapping: dict[str, str] = {}
    with risk_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            species_value = row.get("species_key") or row.get("species_name") or ""
            risk_value = row.get("risk_label") or "unknown"
            species_key = normalize_species_key(species_value)
            if species_key:
                mapping[species_key] = normalize_risk_label(risk_value)
    return mapping


def map_species_to_risk(species_key: str, risk_map: dict[str, str]) -> str:
    return risk_map.get(normalize_species_key(species_key), "unknown")

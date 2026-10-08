from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        p = ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    return cfg

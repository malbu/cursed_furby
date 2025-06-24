import yaml, pathlib

def load():
    path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

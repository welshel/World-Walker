from pathlib import Path
import json
import os

def _load_local_env(config_path):
    for parent in Path(config_path).expanduser().resolve().parents:
        env_path = parent / ".env"
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"'")
            if key and key.replace("_", "").isalnum():
                os.environ.setdefault(key, value)
        break

def load_config(path):
    _load_local_env(path)
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        config = yaml.safe_load(text)
    except ImportError:
        config = json.loads(text)
    def expand_env(value):
        if isinstance(value, dict): return {key: expand_env(item) for key, item in value.items()}
        if isinstance(value, list): return [expand_env(item) for item in value]
        if isinstance(value, str): return os.path.expandvars(value)
        return value
    return expand_env(config)

def expand(value, output, project_root):
    rendered = str(value).format(output=str(output), project_root=str(project_root))
    return Path(os.path.expandvars(rendered)).expanduser().resolve()

from pathlib import Path
import os

ROLLOUT_PREFIX = "rollout"
THINKBRAKE_PREFIX = "thinkbrake"
THINKLESS_PREFIX = "thinkless"

ROOT_DIR = Path(
    os.environ.get("THINKBRAKE_ROOT", Path(__file__).resolve().parent.parent)
)
RESULT_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / THINKBRAKE_PREFIX / "data"

import os, sys
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The dataset defines "today"; never use the system clock.
TODAY = date(2026, 10, 20)

_default = ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026"
DATASET_DIR = Path(os.getenv("DATASET_DIR", _default)).resolve()
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"

# Lets us write: from tools import lookup_order, create_ticket
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))
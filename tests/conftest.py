import sys
from pathlib import Path

# Tests run from project root — add to path explicitly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

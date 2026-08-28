import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from example.quadruped.run import main


if __name__ == "__main__":
    main(lifted_contact_default=True)

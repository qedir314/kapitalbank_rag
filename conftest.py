"""Pytest root config: makes `import kb_rag` work without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pathlib import Path
from typing import Iterable, Tuple
import yaml

def list_markdown_files(content_dir: Path) -> Iterable[Path]:
    for p in content_dir.rglob("*.md"):
        yield p

def read_markdown(path: Path) -> Tuple[dict, str]:
    """
    Returns (metadata_dict, markdown_text)
    - Parses YAML frontmatter if present (--- at top)
    """
    text = path.read_text(encoding="utf-8")
    metadata = {}
    if text.startswith("---"):
        # split frontmatter
        parts = text.split("---", 2)
        if len(parts) >= 3:
            _, fm_text, rest = parts[:3]
            try:
                metadata = yaml.safe_load(fm_text) or {}
            except Exception:
                metadata = {}
            text = rest.lstrip("
")
    return metadata, text

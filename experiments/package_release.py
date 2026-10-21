"""Create and integrity-check a clean ZIP next to the release directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / f"{ROOT.name}.zip"


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in {".git", ".venv", "venv", "__pycache__"} for part in relative.parts):
        return False
    if path.suffix == ".pyc":
        return False
    if len(relative.parts) >= 3 and relative.parts[:2] == ("results", "reproduced"):
        if relative.parts[2].startswith("run_"):
            return False
    return path.is_file()


def main() -> None:
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise SystemExit("SOURCE_MANIFEST.json must report PASS before packaging")

    files = sorted(path for path in ROOT.rglob("*") if should_include(path))
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=7) as archive:
        for path in files:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    with ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise SystemExit(f"ZIP CRC validation failed: {bad_member}")
        member_count = len(archive.namelist())
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(json.dumps({
        "status": "PASS",
        "archive": str(ARCHIVE),
        "members": member_count,
        "bytes": ARCHIVE.stat().st_size,
        "sha256": digest,
    }, indent=2))


if __name__ == "__main__":
    main()

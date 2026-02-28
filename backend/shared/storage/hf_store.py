import os
from datetime import datetime
from typing import Optional
from huggingface_hub import HfApi


def _parse_metadata_from_filename(file_path: str) -> Optional[dict]:
    """Parse session metadata from filenames like session_YYYYMMDD_HHMMSS.csv/parquet."""
    base = os.path.basename(file_path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        dt = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return {
        "filename": base,
        "year": dt.year,
        "month": dt.month,
        "timestamp": dt.isoformat(),
    }


def upload_to_hf(file_path: str, repo_id: str, token: Optional[str] = None) -> str:
    meta = _parse_metadata_from_filename(file_path) or {}
    remote_path = f"{meta.get('filename', os.path.basename(file_path))}"

    api = HfApi(token=token or os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {meta.get('filename', os.path.basename(file_path))}",
    )
    return remote_path

import hashlib, json, time, requests, pathlib, re
from cleaners.common import normalize_whitespace

RAW_ROOT = pathlib.Path("data/raw")

def save_raw_html(html: str, url: str, landmark_id: str, source: str) -> None:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    folder = RAW_ROOT / landmark_id
    folder.mkdir(parents=True, exist_ok=True)

    # 1) raw HTML
    (folder / f"{source}_{h}.html").write_text(html, encoding="utf-8")

    # 2) provenance side-car JSON
    (folder / f"{source}_{h}.meta.json").write_text(
        json.dumps({"url": url,
                    "fetched_at": time.time(),
                    "source": source}, indent=2),
        encoding="utf-8")


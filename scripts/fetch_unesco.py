#!/usr/bin/env python3
"""
Fetch UNESCO World-Heritage-Site pages for every landmark whose row in
manifests/landmarks.yaml has a `unesco_id:` field.

• Saves each page as  data/raw/<landmark_id>/unesco_<hash>.html
• Writes a small unesco_<hash>.meta.json with URL + timestamp
• Prints one line per successful fetch or error
"""

import re, time, yaml, requests
from pathlib import Path
from tqdm import tqdm

from utils.fetch_tools import save_raw_html   # writes the files

MANIFEST  = Path("manifests/landmarks.yaml")
URL_TPL   = "https://whc.unesco.org/en/list/{id}"   # trailing slash = canonical

USER_AGENT = "LandmarkRAG/0.6 (+https://github.com/yourname/landmark-rag)"

def main() -> None:
    landmarks = yaml.safe_load(MANIFEST.read_text("utf-8"))["landmarks"]

    for lm in tqdm(landmarks, desc="UNESCO"):
        uid = lm.get("unesco_id")
        if not uid:                                     # skip if field absent/None
            continue

        url = URL_TPL.format(id=uid)
        print(f"Fetching {url} for {lm['name']} ({lm['id']})")
        try:
            resp = requests.get(url, timeout=15,
                                 headers={"User-Agent": USER_AGENT})
            status = resp.status_code
            html   = resp.text

            if status != 200:
                print(f" ✗ {url} → HTTP {status}")
                continue

            # Simple guard: UNESCO pages with content have <div class="txt">
            if not re.search(r'<div class="txt">', html, re.I):
                print(f" ⚠ {url} looks like a stub; skipping")
                continue

            save_raw_html(html, url, lm["id"], source="unesco")
            print(f" ✓ saved {url} for {lm['name']}")

        except Exception as e:
            print(f" ✗ Error fetching {url}: {e}")

        time.sleep(1)   # be polite to the server

    print("UNESCO data fetching completed.")

if __name__ == "__main__":
    main()

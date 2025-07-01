import yaml, requests, re, time
from utils.fetch_tools import save_raw_html
from pathlib import Path
from tqdm import tqdm

LP_CONFIG = Path("configs/lonelyplanet.yaml")

def main():
    cfg = yaml.safe_load(LP_CONFIG.read_text("utf-8"))
    for lid, urls in cfg.items():
        for url in tqdm(urls, desc=f"Lonely Planet {lid}"):
            try:
                html = requests.get(url, timeout=15,
                                    headers={"User-Agent":"LandmarkRAG/0.7"}).text
                if len(re.sub(r"<[^>]+>", " ", html)) < 200:
                    print(f"⚠ {lid}: very short page; skipping")
                    continue
                save_raw_html(html, url, lid, source="lonelyplanet")
            except Exception as e:
                print("  ✗", url, e)
            time.sleep(1)  # be polite to the server
    print("Lonely Planet data fetching completed.")

if __name__ == "__main__":
    main()
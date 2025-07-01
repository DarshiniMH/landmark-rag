import yaml
from pathlib import Path

MANIFEST_FILE = Path("manifests/landmarks.yaml")
CONFIG_DIR = Path("configs")
OUTPUT_FILE = CONFIG_DIR/"lonelyplanet.yaml"

PLACEHOLDER_URL = "PASTE_RELEVANT_URLS_HERE"

def main():
    print("generating Lonely Planet crawler config...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        manifest_data = yaml.safe_load(MANIFEST_FILE.read_text())
        landmarks = manifest_data["landmarks"]
    except FileNotFoundError:
        print(f"Error: The file {MANIFEST_FILE} does not exist.")
        return
    
    source_data = {}

    for landmark in landmarks:
        landmark_id = landmark["id"]
        source_data[landmark_id] = [PLACEHOLDER_URL]
    
    with OUTPUT_FILE.open("w", encoding = "utf_8") as f:
        yaml.safe_dump(source_data, f, sort_keys = False, allow_unicode=True)
    
    print(f"wrote {OUTPUT_FILE} of length {len(landmarks)}")

if __name__ == "__main__":
    main()
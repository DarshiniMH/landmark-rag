"""
Reads the master configuration file and generates a crawler configuration file 
for fetching Wikipedia data based on the landmarks defined in the manifest.
"""

import yaml
from pathlib import Path

# ── Config --

MANIFEST_FILE = Path("manifests/landmarks.yaml")
CONFIG_DIR = Path("configs")

# THis is the template that will be written to each file
CONFIG_TEMPLATE = r"""\
# Configuration for scraping: {landmark_name}
# Canonical ID: {landmark_id}

# The starting point(s) for the crawl.
# TODO: Manually find and paste the best starting URL (e.g., a "History" or "About" page).
seed_urls:
  - "PASTE_PRIMARY_URL_HERE"

# Regex patterns to define which URLs the crawler is ALLOWED to visit.
# TODO: Manually refine this pattern based on the seed URL.
allow_patterns:
  - "PASTE_REGEX_PATTERN_HERE"

# These generic patterns are usually fine and don't need to be changed.
deny_patterns:
  - \.(jpg|jpeg|png|gif|svg|mp4|webm|pdf|zip)$
  - /cart$
  - /checkout$
  - /contact
  - /legal

# Automatically populated list of aliases for content validation.
aliases:
{aliases_list}

# This rarely needs to be changed.
crawl_depth: 2
"""

def main():
    print("Generating crawler configuration files...")

    # create the output "configs" directory if it doesnot exist
    CONFIG_DIR.mkdir(parents = True, exist_ok=True)

    try:
        manifest_data = yaml.safe_load(MANIFEST_FILE.read_text())
        landmarks = manifest_data["landmarks"]
    except FileNotFoundError:
        print(f"Error: The file {MANIFEST_FILE} does not exist.")
        return
    
    configs_created = 0
    for landmark in landmarks:
        landmark_id = landmark["id"]
        landmark_name = landmark["name"]

        # combime main name and aliases in one list
        aliases = [landmark_name] + landmark.get("aliases", [])

        # format aliases for yaml template
        aliases_yaml = "\n".join(f"  - \"{alias}\"" for alias in aliases)

        # fill the template with the landmark data
        config_content = CONFIG_TEMPLATE.format(
            landmark_name=landmark_name,
            landmark_id=landmark_id,
            aliases_list=aliases_yaml
        )

        output_path = CONFIG_DIR / f"{landmark_id}.yaml"

        # Write new config file
        output_path.write_text(config_content, encoding = "utf-8")
        print(f"Created config for {landmark_name} ({landmark_id}) at {output_path}")

if __name__ == "__main__":
    main()
    print("Done generating crawler configuration files.")

#import necessary libraries
import yaml
import wikipedia
import json
import os
from pathlib import Path
import datetime
import time

# Define where to find the input file and where to save the output file
MANIFEST_FILE = "manifests/landmarks.yaml"
OUTPUT_DIR = Path("data/raw")
FRESHNESS_DAYS = 7

# --Main logic--

print("Starting to fetch Wikipedia data...")

# create output directory if it isnt present
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#open and read list of landmarks from the manifest file
try:
    with open(MANIFEST_FILE, 'r') as f:
        landmarks_list = yaml.safe_load(f)["landmarks"]
except FileNotFoundError:
    print(f"Error: The file {MANIFEST_FILE} does not exist.")
    exit()

# Iterate through each landmark in the list
for landmark in landmarks_list:
    try:
        #get uique id and wikipedia title for the landmark
        landmark_id = landmark["id"]
        wiki_title = landmark["wikipedia_title"]

        print(f"Processing '{landmark_id}' with Wikipedia title '{wiki_title}'...")

        landmark_dir = OUTPUT_DIR/landmark_id

        # Create a directory for the landmark if it doesn't exist
        landmark_dir.mkdir(exist_ok=True)

        # Define the output file path
        output_filepath = landmark_dir/"wikipedia.json"

        # Check if the file already exists and if it's fresh
        if output_filepath.exists():
            try:
                # Read the timestamp from the existing file
                json_data = json.loads(output_filepath.read_text(encoding='utf-8'))
                fetched_at_str = json_data["fetched_at"]

                # convert the timestamp into datetime object
                fetched_at_dt = datetime.datetime.fromisoformat(fetched_at_str)

                #get current time
                now_dt = datetime.datetime.now(datetime.timezone.utc)

                # calculate age of the file in days
                age_days = (now_dt - fetched_at_dt).days
                if age_days < FRESHNESS_DAYS:
                    print(f"Skipping '{landmark_id}' as it is fresh (age: {age_days} days).")
                    continue
            except(KeyError, json.JSONDecodeError):
                print(f"Error reading the timestamp from '{landmark_id}'. The file may be corrupted.")

        # Fetch the Wikipedia page for the landmark
        page = wikipedia.page(wiki_title, auto_suggest=False, redirect = True)

        # prepare the data to be saved
        data_to_save = {
            "id": landmark_id,
            "title" : page.title,
            "url": page.url,
            "content": page.content,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # write data to file. indent=2 makes json file human redable
        with output_filepath.open('w', encoding = 'utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"Saved data for '{landmark_id}' to '{output_filepath}'.")

    
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for '{landmark_id}': {e}.")
        # You can choose to handle this case differently, e.g., log it or skip it.
    except wikipedia.exceptions.PageError as e:
        print(f"Page error for '{landmark_id}': {e}.")
        # You can choose to handle this case differently, e.g., log it or skip it.
    except KeyError as e:
        print(f"Key error for '{landmark_id}': {e}.")
        # You can choose to handle this case differently, e.g., log it or skip it.
    except Exception as e:
        print(f"An unexpected error occurred for '{landmark_id}': {e}.")
        # You can choose to handle this case differently, e.g., log it or skip it.

    time.sleep(1)  # Sleep for 1 second to avoid hitting the API too hard

print("Finished fetching Wikipedia data.")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from cleaners import get_cleaner
from cleaners.wikipedia import sections_with_context
from pathlib import Path
import json, uuid, pathlib


RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")

#Initializing the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 400,
    chunk_overlap = 50,
    separators    = ["\n\n", ". ", " ", ""],  # paragraph → sentence → word → char
)

def process_one_file(id_code: str, source: str, path: Path, out_f):
    raw = json.loads(path.read_text())
    text = get_cleaner(source)(raw["content"])

    if source == "wikipedia":
        sections = sections_with_context(text)
    else:
        sections = [("Full text", text)]

    for title, section in sections:
        sections_prefixed = f"{title}\n{section}"
        for chunk in splitter.split_text(sections_prefixed):
            out_f.write(json.dumps({
                "chunk_id"      : str(uuid.uuid4()),
                "landmark_id"   : id_code,
                "source"        : source,
                "section:"      : title,    
                "origin_url"    : raw["url"],
                "text"          : chunk,
                "fetched_at"    : raw["fetched_at"],
            }, ensure_ascii=False) + "\n")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding = "utf-8") as out_f:
        for lm_dir in RAW_DIR.iterdir():
            if not lm_dir.is_dir():
                continue
            id_code = lm_dir.name
            wiki_file = lm_dir/"wikipedia.json"
            if wiki_file.exists():
                process_one_file(id_code, "wikipedia", wiki_file, out_f)
                print(f"Processed {id_code} from Wikipedia.")
    print(f"Processed chunks saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
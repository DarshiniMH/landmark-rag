import re, unicodedata

def normalize_whitespace(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", txt)
    txt = re.sub(r"\r\n?", "\n", txt)        # Windows → Unix newlines
    txt = re.sub(r"\n{3,}", "\n\n", txt)     # collapse blank lines
    txt = re.sub(r"[ \t]{2,}", " ", txt)     # collapse spaces/tabs
    return txt.strip()
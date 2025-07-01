from .wikipedia import clean_wikipedia_text
from .html_generic import clean_html_text

# Map source tag -> cleaner function
SOURCE_CLEANERS = {
    "wikipedia": clean_wikipedia_text,
    "html" : clean_html_text,
}

def get_cleaner(source: str):
    try:
        return SOURCE_CLEANERS[source]
    except KeyError:
        raise ValueError(f"No cleaner registered for source='{source}'")

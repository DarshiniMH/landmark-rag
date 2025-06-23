from .wikipedia import clean_wikipedia_text

# Map source tag -> cleaner function
SOURCE_CLEANERS = {
    "wikipedia": clean_wikipedia_text,
}

def get_cleaner(source: str):
    try:
        return SOURCE_CLEANERS[source]
    except KeyError:
        raise ValueError(f"No cleaner registered for source='{source}'")

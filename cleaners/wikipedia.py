"""
Adapter for cleaning Wikipedia text. 

Public API:
clean_wikipedia_text(raw_wiki_content: str)->str
#   Returns cleaned Wikipedia text.

sections_with_context(cleaned_wiki_text: str) -> list[tuple[str, str]]
    split the cleaned Wikipedia text into sections with context.
"""

# from __future__ import annotations is a modern feature that helps with type hinting
# especially for complex types

from __future__ import annotations

import re
from typing import List, Tuple

from .common import normalize_whitespace

# Configurable constants
# This section defines al the settings and patterns in one place. 
# A simple list of section headers that usually mark the end of the useful,
# relevant content in a Wikipedia article.

FOOTER_HEADERS = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Notes",
    "Bibliography",
    "Sources",
    "Related pages",
    "Citations",
    "Footnotes",
    "Works cited",
]

# 're.complile' pre-builds the regular expression pattern for a significant speed
# improvement. 

FOOTER_PATTERN = re.compile(
    r"\n==+\s*(?:"
    + "|".join(re.escape(h) for h in FOOTER_HEADERS)
    + r")\s*==+\s*\n",
    flags=re.IGNORECASE,
)

INLINE_CITATION_PATTERN = re.compile(r"\[[^\]]*]")  # [23], [citation needed]

TABLE_LINE_PATTERN = re.compile(r"^\s*(?:\{\{|\|\}|\{|}\||\|)", flags=re.MULTILINE)

# Low- Level helper functions

def _remove_footer_sections(text:str) ->str:
    """
    Finds the first occurrence of a footer section in the text and removes it.
    """
    match = FOOTER_PATTERN.search(text)
    # "return the text up to the start of the match, if a match `m` was found.
    # Otherwise (`else`), return the original text unchanged."
    return text[: match.start()] if match else text

def _remove_inline_citations(text:str) ->str:
    """
    Removes inline citations from the text.
    """
    # .sub(replacement, original_text) finds all matches and substitutes them.
    return INLINE_CITATION_PATTERN.sub("", text)

def _remove_tables(text:str) ->str:
    """
    Removes tables from the text.
    """
    return TABLE_LINE_PATTERN.sub("", text)


# Public cleaning pipeline
def clean_wikipedia_text(raw: str) ->str:
    """
    This is the public function. It orchestrates the cleaning process
    by calling private helper functions. 
    Low level helper function cleans the raw wikipedia data to remove
    unwanted sections, inline citations, tables and normalized whitespaces.
    """
    # Remove unwanted sections
    cleaned = _remove_footer_sections(raw)
    # Remove tables
    cleaned =  _remove_tables(cleaned)
    # Remove inline citations
    cleaned = _remove_inline_citations(cleaned)
    # Normalize whitespace
    cleaned = normalize_whitespace(cleaned)

    return cleaned


# Section aware pre-chunking
# This section is responsible for splitting the cleaned text into sections
_SECTION_SPLIT_REGEX = re.compile(r'\n(={2,6})\s*([^=]+?)\s*\1\s*\n')


def sections_with_context(cleaned: str) -> List[Tuple(str, str)]:
    """
    This function restructures the document into a list of (title, content)
    pairs, which is a very effective strategy for context-aware chunking.

    Returns [(section_title, section_body), …] where the first element is the
    introduction.  Titles come from `== Heading ==` markers (2-6 '=' chars).

    Example:
        >>> secs = sections_with_context(cleaned)
        >>> secs[0][0]    # "Introduction"
        >>> secs[1][0]    # "History"
    """
    chunks = _SECTION_SPLIT_REGEX.split(cleaned)

    if not chunks:
        return [("Introduction", cleaned)]
    
    # The first element of the split list is always the text before the first header.
    sections: list[Tuple[str, str]] = [("Introduction", chunks[0].strip())]

    for i in range(1, len(chunks) -2, 3):
         title = chunks[i+1].strip()
         body = chunks[i+2].strip()
         if body:
             sections.append((title, body))

    return sections


# what other modules import from this mocule:
__all__ = [
    "clean_wikipedia_text",
    "sections_with_context",
]

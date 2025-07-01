from __future__ import annotations
from .common import normalize_whitespace
import re
from bs4 import BeautifulSoup
from trafilatura import extract

UNWANTED_PATTERNS = [
    r"(?i)cookies? settings?",
    r"(?i)subscribe to our newsletter",
    r"Share this (page|article)"
]
def clean_html_text(raw_html: str) ->str:
    text = extract(raw_html, include_comments = False, favor_precision = True) 
    if not text:
        soup = BeautifulSoup(raw_html, "lxml")
        return soup.get_text(" ", strip = True)
    
    for pat in UNWANTED_PATTERNS:
        text = re.sub(pat, " ", text)
    
        return normalize_whitespace(text)
__all__ = ["clean_html_text"]   
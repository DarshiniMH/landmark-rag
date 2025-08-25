from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 710
CHUNK_OVERLAP = 135
SEPARATORS = ["\n\n", ". ", " ", ""]

def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES = Path("data/articles")


def chunk_text(text, size=500, overlap=50):
    """Split text into pieces of about `size` characters.
    Each piece repeats the last `overlap` characters of the previous one,
    so a sentence cut at a boundary still appears whole somewhere."""
    chunks = []
    start = 0
    while True:
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")

    # Start fresh each time so re-running never creates duplicates
    try:
        client.delete_collection("help_articles")
    except Exception:
        pass
    collection = client.create_collection("help_articles")

    ids, docs, metas = [], [], []
    for path in sorted(ARTICLES.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        for n, chunk in enumerate(chunk_text(text)):
            if not chunk.strip():
                continue
            ids.append(f"{path.stem}-{n}")
            docs.append(f"{title}: {chunk}") #contextual chunk headers
            metas.append({"source": path.name})

    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metas,
                   embeddings=embeddings)

    n_articles = len({m["source"] for m in metas})
    print(f"Stored {len(docs)} chunks from {n_articles} articles")


if __name__ == "__main__":
    main()
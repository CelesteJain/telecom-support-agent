import sys

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection(
    "help_articles"
)


def search(question, top_k=4, candidates=20, use_reranker=True, min_score=0.0):
    """Return up to `top_k` chunks. With the re-ranker on, drop any chunk
    scoring below `min_score`, so weak matches are not passed to the model."""
    query_vector = embedder.encode([question]).tolist()
    found = collection.query(query_embeddings=query_vector,
                             n_results=candidates)

    results = [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(found["documents"][0], found["metadatas"][0])
    ]

    if use_reranker:
        scores = reranker.predict([(question, r["text"]) for r in results])
        for r, score in zip(results, scores):
            r["score"] = float(score)
        results.sort(key=lambda r: r["score"], reverse=True)
        results = [r for r in results if r["score"] >= min_score]

    return results[:top_k]


def show(results):
    for r in results:
        preview = r["text"][:100].replace("\n", " ")
        score = f"{r['score']:.2f} " if "score" in r else ""
        print(f"  {score}[{r['source']}] {preview}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "my router has no internet"
    print("--- vector search only ---")
    show(search(question, use_reranker=False))
    print("--- with re-ranker ---")
    show(search(question))
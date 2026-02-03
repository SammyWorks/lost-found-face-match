import numpy as np

def cosine_similarity(a, b):
    # a and b are already normalized
    return np.dot(a, b)

def find_top_matches(query_embedding, db_embeddings, db_metadata, top_k=3):
    results = []

    for idx, emb in enumerate(db_embeddings):
        score = cosine_similarity(query_embedding, emb)
        results.append((score, db_metadata[idx]))

    # Higher score = more similar
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

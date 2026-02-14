from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def embed_texts(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, query):
        return self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(
            documents,
            scores
        ):

            result = document.copy()

            result["rerank_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
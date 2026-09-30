# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):

        self.bm25 = None
        self.chunks = []

    def build(self, chunks):

        self.chunks = chunks

        tokenized_chunks = [

            chunk["text"].lower().split()

            for chunk in chunks

        ]

        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    def search(
        self,
        query,
        top_k=10,
        paper_id=None
    ):

        if self.bm25 is None:
            return []

        tokenized_query = (
            query.lower().split()
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        results = []

        for index in scores.argsort()[::-1]:

            chunk = self.chunks[index]

            if (
                paper_id
                and chunk["paper_id"] != paper_id
            ):
                continue

            results.append(
                {
                    **chunk,
                    "score": float(
                        scores[index]
                    )
                }
            )

            if len(results) >= top_k:
                break

        return results
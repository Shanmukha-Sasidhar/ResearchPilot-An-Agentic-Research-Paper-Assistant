from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever


class Retriever:

    def __init__(self):

        self.vector_store = VectorStore()

        self.bm25 = BM25Retriever()

    def build(self, chunks):

        self.vector_store.create(
            chunks
        )

        all_chunks = (
            self.vector_store.get_all_chunks()
        )

        self.bm25.build(
            all_chunks
        )

    def retrieve(
        self,
        query,
        top_k=5,
        paper_id=None
    ):

        semantic_results = (
            self.vector_store.search(
                query,
                top_k=10,
                paper_id=paper_id
            )
        )

        bm25_results = (
            self.bm25.search(
                query,
                top_k=10,
                paper_id=paper_id
            )
        )

        combined = {}

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            chunk_id = (
                f"{result['paper_id']}_"
                f"{result['chunk_id']}"
            )

            combined[chunk_id] = {
                **result,
                "semantic_rank": rank,
                "bm25_rank": None
            }

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk_id = (
                f"{result['paper_id']}_"
                f"{result['chunk_id']}"
            )

            if chunk_id not in combined:

                combined[chunk_id] = {
                    **result,
                    "semantic_rank": None,
                    "bm25_rank": rank
                }

            else:

                combined[
                    chunk_id
                ]["bm25_rank"] = rank

        for result in combined.values():

            score = 0

            if result["semantic_rank"]:
                score += (
                    1 /
                    (60 + result["semantic_rank"])
                )

            if result["bm25_rank"]:
                score += (
                    1 /
                    (60 + result["bm25_rank"])
                )

            result["hybrid_score"] = score

        results = sorted(
            combined.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return results[:top_k]
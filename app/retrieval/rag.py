from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.generation.llm import ask_llm


class RAGPipeline:

    def __init__(self):

        self.retriever = Retriever()

        self.reranker = Reranker()

    def build(self, chunks):

        self.retriever.build(chunks)

    def ask(
        self,
        question,
        top_k=5
    ):

        candidates = (
            self.retriever.retrieve(
                question,
                top_k=10
            )
        )

        results = (
            self.reranker.rerank(
                question,
                candidates,
                top_k=top_k
            )
        )

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {index}

Paper:
{result['paper_name']}

Page:
{result['page_number']}

Content:
{result['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        answer = ask_llm(
            question=question,
            context=context
        )

        return {
            "answer": answer,
            "sources": results
        }
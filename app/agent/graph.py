from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.retrieval.rag import RAGPipeline
from app.generation.llm import ask_llm


class AgentState(TypedDict):

    question: str

    search_query: str

    context: str

    results: list

    answer: str

    attempts: int

    chat_history: list


class ResearchAgent:

    def __init__(self, chunks):

        self.rag = RAGPipeline()

        self.rag.build(chunks)

        self.graph = self._build_graph()

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    def _retrieve(self, state):

        query = state["search_query"]

        results = self.rag.retriever.retrieve(
            query,
            top_k=10
        )

        reranked = self.rag.reranker.rerank(
            query,
            results,
            top_k=5
        )

        context_parts = []

        for result in reranked:

            context_parts.append(
                f"""
[Page {result['page_number']}]

{result['text']}
"""
            )

        context = "\n\n".join(context_parts)

        return {
            "results": reranked,
            "context": context,
            "attempts": state["attempts"] + 1,
        }

    # --------------------------------------------------
    # CHECK RETRIEVED CONTEXT
    # --------------------------------------------------

    def _check_context(self, state):

        context = state["context"]

        if len(context.strip()) < 200:

            return "rewrite"

        return "answer"

    # --------------------------------------------------
    # CONTEXTUALIZE QUESTION
    # --------------------------------------------------

    def _contextualize_question(self, state):

        history = state["chat_history"]

        # First question -> no need for contextualization

        if not history:

            return {
                "search_query": state["question"]
            }

        # Use the last 4 conversation turns

        history_text = "\n".join(
            [
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}"
                for item in history[-4:]
            ]
        )

        prompt = f"""
Rewrite the latest user question so that
it is understandable without conversation history.

Conversation:

{history_text}

Latest question:

{state['question']}

Return ONLY the rewritten question.
"""

        rewritten = ask_llm(
            question=prompt,
            context=""
        )

        return {
            "search_query": rewritten.strip()
        }

    # --------------------------------------------------
    # QUERY REWRITING
    # --------------------------------------------------

    def _rewrite_query(self, state):

        question = state["question"]

        rewrite_prompt = f"""
Rewrite the following research-paper question
into a better search query.

The query should contain the important technical
concepts needed for retrieval.

Original question:

{question}

Return only the rewritten search query.
"""

        rewritten_query = ask_llm(
            question=rewrite_prompt,
            context=""
        )

        return {
            "search_query": rewritten_query.strip()
        }

    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    def _generate_answer(self, state):

        answer = ask_llm(
            question=state["question"],
            context=state["context"]
        )

        return {
            "answer": answer
        }

    # --------------------------------------------------
    # BUILD LANGGRAPH
    # --------------------------------------------------

    def _build_graph(self):

        graph = StateGraph(AgentState)

        # Nodes

        graph.add_node(
            "contextualize",
            self._contextualize_question
        )

        graph.add_node(
            "retrieve",
            self._retrieve
        )

        graph.add_node(
            "rewrite",
            self._rewrite_query
        )

        graph.add_node(
            "answer",
            self._generate_answer
        )

        # --------------------------------------------------
        # GRAPH FLOW
        # --------------------------------------------------

        # START
        #   ↓
        # contextualize
        #   ↓
        # retrieve
        #   ↓
        # check context
        #   ↓
        # ┌───────────────┐
        # │               │
        # answer        rewrite
        #                  ↓
        #               retrieve

        graph.add_edge(
            START,
            "contextualize"
        )

        graph.add_edge(
            "contextualize",
            "retrieve"
        )

        graph.add_conditional_edges(
            "retrieve",
            self._check_context,
            {
                "answer": "answer",
                "rewrite": "rewrite",
            }
        )

        graph.add_edge(
            "rewrite",
            "retrieve"
        )

        graph.add_edge(
            "answer",
            END
        )

        return graph.compile()

    # --------------------------------------------------
    # PUBLIC ASK METHOD
    # --------------------------------------------------

    def ask(
        self,
        question,
        chat_history=None
    ):

        if chat_history is None:

            chat_history = []

        initial_state = {

            "question": question,

            "search_query": question,

            "context": "",

            "results": [],

            "answer": "",

            "attempts": 0,

            "chat_history": chat_history,
        }

        result = self.graph.invoke(
            initial_state
        )

        return {
            "answer": result["answer"],
            "sources": result["results"],
        }
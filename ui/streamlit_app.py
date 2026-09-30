import sys
from pathlib import Path

# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# IMPORTS
# ==================================================

import os
import streamlit as st

from main import ResearchPaperApplication


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="ResearchPilot — An Agentic Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)


# ==================================================
# APPLICATION TITLE
# ==================================================

st.title(
    "📚 Research Paper Intelligence Agent"
)

st.caption(
    "Agentic RAG with hybrid retrieval, "
    "cross-encoder reranking and conversational memory."
)


# ==================================================
# SESSION STATE
# ==================================================

if "application" not in st.session_state:

    st.session_state.application = (
        ResearchPaperApplication()
    )


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def display_sources(sources):

    if not sources:
        return

    st.markdown("### 📚 Sources")

    for index, source in enumerate(
        sources,
        start=1
    ):

        paper_name = source.get(
            "paper_name",
            "Unknown paper"
        )

        page_number = source.get(
            "page_number",
            "?"
        )

        rerank_score = source.get(
            "rerank_score",
            0
        )

        with st.expander(
            f"Source {index} — "
            f"{paper_name} — "
            f"Page {page_number}"
        ):

            st.caption(
                f"Rerank score: "
                f"{rerank_score:.4f}"
            )

            st.write(
                source.get(
                    "text",
                    ""
                )
            )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Research Papers")

    # --------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a research paper",
        type=["pdf"]
    )

    if uploaded_file:

        os.makedirs(
            "data/papers",
            exist_ok=True
        )

        file_path = os.path.join(
            "data/papers",
            uploaded_file.name
        )

        # Save PDF

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        # --------------------------------------------------
        # PROCESS BUTTON
        # --------------------------------------------------

        if st.button(
            "⚙️ Process Paper",
            use_container_width=True
        ):

            with st.spinner(
                "Processing research paper..."
            ):

                try:

                    paper = (
                        st.session_state
                        .application
                        .process_paper(
                            file_path
                        )
                    )

                    # Clear previous conversation

                    st.session_state.chat_history = []

                    st.success(
                        "Paper processed successfully!"
                    )

                    st.info(
                        f"""
**Paper:** {paper['paper_name']}

**Pages:** {paper['pages']}

**Chunks:** {paper['chunks']}
"""
                    )

                except Exception as e:

                    st.error(
                        "Error while processing paper:"
                    )

                    st.exception(e)

    # --------------------------------------------------
    # DIVIDER
    # --------------------------------------------------

    st.divider()

    # --------------------------------------------------
    # INDEXED PAPERS
    # --------------------------------------------------

    st.subheader(
        "📚 Indexed Papers"
    )

    papers = (
        st.session_state
        .application
        .get_papers()
    )

    if papers:

        for paper_id, paper in papers.items():

            st.markdown(
                f"**📄 {paper['paper_name']}**"
            )

            st.caption(
                f"{paper['pages']} pages • "
                f"{paper['chunks']} chunks"
            )

            st.divider()

    else:

        st.caption(
            "No papers processed yet."
        )

    # --------------------------------------------------
    # CURRENT PAPER
    # --------------------------------------------------

    current_paper = (
        st.session_state
        .application
        .get_current_paper()
    )

    if current_paper:

        st.subheader(
            "Current Paper"
        )

        st.info(
            f"""
**{current_paper['paper_name']}**

Pages: {current_paper['pages']}

Chunks: {current_paper['chunks']}
"""
        )

    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ==================================================
# NO PAPER PROCESSED
# ==================================================

if not papers:

    st.info(
        "👈 Upload a research paper from the sidebar "
        "and click **Process Paper** to begin."
    )

    st.stop()


# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state.chat_history:

    # --------------------------------------------------
    # USER
    # --------------------------------------------------

    with st.chat_message("user"):

        st.write(
            message["question"]
        )

    # --------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------

    with st.chat_message("assistant"):

        st.write(
            message["answer"]
        )

        display_sources(
            message.get(
                "sources",
                []
            )
        )


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about the research paper..."
)


# ==================================================
# HANDLE QUESTION
# ==================================================

if question:

    # --------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------

    with st.chat_message("user"):

        st.write(
            question
        )

    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching, reranking and reasoning..."
        ):

            try:

                result = (
                    st.session_state
                    .application
                    .ask(
                        question=question,
                        chat_history=(
                            st.session_state
                            .chat_history
                        )
                    )
                )

                answer = result.get(
                    "answer",
                    "No answer generated."
                )

                sources = result.get(
                    "sources",
                    []
                )

                # ------------------------------------------
                # ANSWER
                # ------------------------------------------

                st.write(
                    answer
                )

                # ------------------------------------------
                # SOURCES
                # ------------------------------------------

                display_sources(
                    sources
                )

                # ------------------------------------------
                # SAVE CHAT
                # ------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                st.error(
                    "Error while answering question:"
                )

                st.exception(e)
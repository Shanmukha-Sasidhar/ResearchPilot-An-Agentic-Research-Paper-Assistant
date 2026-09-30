from pathlib import Path
import hashlib

from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import create_chunks
from app.agent.graph import ResearchAgent


class ResearchPaperApplication:

    def __init__(self):
        self.agent = None
        self.papers = {}
        self.current_paper = None

    # ==================================================
    # PROCESS PAPER
    # ==================================================

    def process_paper(self, file_path):

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {file_path}"
            )

        paper_name = file_path.name

        # Create unique paper ID from file contents

        with open(file_path, "rb") as file:
            file_bytes = file.read()

        paper_id = hashlib.md5(
            file_bytes
        ).hexdigest()[:12]

        # --------------------------------------------------
        # PARSE PDF
        # --------------------------------------------------

        pages = parse_pdf(
            str(file_path)
        )

        if not pages:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        # --------------------------------------------------
        # CREATE CHUNKS
        # --------------------------------------------------

        chunks = create_chunks(
            pages=pages,
            paper_id=paper_id,
            paper_name=paper_name
        )

        if not chunks:
            raise ValueError(
                "No chunks were created from the PDF."
            )

        # --------------------------------------------------
        # STORE PAPER INFORMATION
        # --------------------------------------------------

        self.papers[paper_id] = {
            "paper_id": paper_id,
            "paper_name": paper_name,
            "file_path": str(file_path),
            "pages": len(pages),
            "chunks": len(chunks),
        }

        # --------------------------------------------------
        # CREATE / UPDATE AGENT
        # --------------------------------------------------

        if self.agent is None:

            self.agent = ResearchAgent(
                chunks
            )

        else:

            self.agent.rag.build(
                chunks
            )

        self.current_paper = paper_id

        return self.papers[paper_id]

    # ==================================================
    # ASK QUESTION
    # ==================================================

    def ask(
        self,
        question,
        chat_history=None
    ):

        if self.agent is None:
            raise ValueError(
                "No research paper has been processed yet."
            )

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if chat_history is None:
            chat_history = []

        return self.agent.ask(
            question=question,
            chat_history=chat_history
        )

    # ==================================================
    # GET CURRENT PAPER
    # ==================================================

    def get_current_paper(self):

        if self.current_paper is None:
            return None

        return self.papers.get(
            self.current_paper
        )

    # ==================================================
    # GET ALL PAPERS
    # ==================================================

    def get_papers(self):

        return self.papers


# ======================================================
# COMMAND LINE MODE
# ======================================================

def main():

    application = ResearchPaperApplication()

    file_path = "data/papers/rag.pdf"

    try:

        paper = application.process_paper(
            file_path
        )

        print("\n" + "=" * 70)
        print("RESEARCH PAPER PROCESSED")
        print("=" * 70)

        print(
            f"Paper  : {paper['paper_name']}"
        )

        print(
            f"Pages  : {paper['pages']}"
        )

        print(
            f"Chunks : {paper['chunks']}"
        )

        print("\nResearch Paper Agent is ready.")

        while True:

            question = input(
                "\nAsk a question "
                "(type 'exit' to quit): "
            )

            if question.lower().strip() == "exit":
                break

            try:

                result = application.ask(
                    question
                )

                print("\n" + "=" * 70)
                print("ANSWER")
                print("=" * 70)

                print(
                    result["answer"]
                )

                print("\n" + "=" * 70)
                print("SOURCES")
                print("=" * 70)

                for index, source in enumerate(
                    result["sources"],
                    start=1
                ):

                    print(
                        f"\nSource {index}"
                    )

                    print(
                        f"Paper: "
                        f"{source.get('paper_name', 'Unknown')}"
                    )

                    print(
                        f"Page: "
                        f"{source.get('page_number', '?')}"
                    )

                    print(
                        f"Rerank score: "
                        f"{source.get('rerank_score', 0):.4f}"
                    )

                    print(
                        source["text"][:500]
                    )

            except Exception as e:

                print(
                    f"\nError while answering question: {e}"
                )

    except Exception as e:

        print(
            f"\nError while processing paper: {e}"
        )


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages, paper_id, paper_name):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk_text in page_chunks:

            chunks.append(
                {
                    "paper_id": paper_id,
                    "paper_name": paper_name,
                    "chunk_id": chunk_id,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                }
            )

            chunk_id += 1

    return chunks
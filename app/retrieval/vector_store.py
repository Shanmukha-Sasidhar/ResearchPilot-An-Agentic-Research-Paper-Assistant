import chromadb

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:

    def __init__(self):

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        self.client = chromadb.PersistentClient(
            path="data/processed/chroma_db"
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="research_papers"
            )
        )

    def create(self, chunks):

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        ids = [
            f"{chunk['paper_id']}_{chunk['chunk_id']}"
            for chunk in chunks
        ]

        metadatas = [

            {
                "paper_id": chunk["paper_id"],
                "paper_name": chunk["paper_name"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            }

            for chunk in chunks
        ]

        self.collection.upsert(

            ids=ids,

            documents=texts,

            embeddings=embeddings,

            metadatas=metadatas,
        )

    def search(
        self,
        query,
        top_k=10,
        paper_id=None
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
        }

        if paper_id:

            kwargs["where"] = {
                "paper_id": paper_id
            }

        results = self.collection.query(
            **kwargs
        )

        retrieved_chunks = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            retrieved_chunks.append(
                {
                    "text": document,
                    "paper_id": metadata["paper_id"],
                    "paper_name": metadata["paper_name"],
                    "page_number": metadata["page_number"],
                    "chunk_id": metadata["chunk_id"],
                    "score": 1 - distance,
                }
            )

        return retrieved_chunks

    def get_all_chunks(self):

        results = self.collection.get()

        chunks = []

        for document, metadata in zip(
            results["documents"],
            results["metadatas"]
        ):

            chunks.append(
                {
                    "text": document,
                    "paper_id": metadata["paper_id"],
                    "paper_name": metadata["paper_name"],
                    "page_number": metadata["page_number"],
                    "chunk_id": metadata["chunk_id"],
                }
            )

        return chunks
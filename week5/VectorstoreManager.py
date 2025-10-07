import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorstoreManager:
    def __init__(self, embedding_model, db_base_name, chunk_size):
        """
        Args:
            embedding_model (str): The Google Generative AI embedding model to use.
            db_base_name (str): Base name for persistent vectorstore directories.
            chunk_size (int): Number of characters per chunk.
        """
        self.embedding_model = embedding_model
        self.db_base_name = db_base_name
        self.chunk_size = chunk_size
        self.vectorstores = []

        # Calculate batch size based on chunk_size (~4 chars/token, max 20k tokens)
        self.tokens_per_chunk = max(1, self.chunk_size // 4)
        self.chunks_per_batch = max(1, 20000 // self.tokens_per_chunk)
        print(f"Each chunk ~{self.tokens_per_chunk} tokens, max {self.chunks_per_batch} chunks per batch (~20k tokens)")

    def _get_embeddings(self):
        """Get a new embeddings instance using next API key."""
        import sys
        sys.path.append('/run/media/sijanpaudel/New Volume/New folder/llm_engineering')
        from key_utils import get_next_key

        api_key, key_name = get_next_key()
        os.environ["GOOGLE_API_KEY"] = api_key
        print(f"Using API key: {key_name}")
        return GoogleGenerativeAIEmbeddings(model=self.embedding_model)

    def _merge_vectorstores(self, target_store, source_store):
        """Merge source_store into target_store."""
        data = source_store.get(include=["metadatas", "documents", "embeddings"])
        docs, metas, embs, ids = data['documents'], data['metadatas'], data['embeddings'], data['ids']

        print(f"Merging {len(docs)} docs from {source_store._collection.name} into {target_store._collection.name}")
        target_store._collection.add(
            embeddings=embs,
            documents=docs,
            metadatas=metas,
            ids=ids
        )
        return target_store

    def create_vectorstores(self, chunks):
        """
        Split chunks into batches, create vectorstores for each batch using rotating API keys,
        and automatically merge all batches into a single final vectorstore.
        
        Returns:
            Chroma: The final merged vectorstore.
        """
        self.vectorstores = []

        for i in range(0, len(chunks), self.chunks_per_batch):
            batch = chunks[i:i + self.chunks_per_batch]
            db_name = f"{self.db_base_name}_{i // self.chunks_per_batch}"

            embeddings = self._get_embeddings()

            # Delete existing collection if any
            if os.path.exists(db_name):
                Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

            # Create vectorstore
            vs = Chroma.from_documents(documents=batch, embedding=embeddings, persist_directory=db_name)
            print(f"✅ Vectorstore {db_name} created with {vs._collection.count()} documents")
            self.vectorstores.append(vs)

        # Automatically merge all vectorstores into the first one
        final_store = self.vectorstores[0]
        for vs in self.vectorstores[1:]:
            final_store = self._merge_vectorstores(final_store, vs)

        print(f"🎉 Final merged vectorstore has {final_store._collection.count()} documents")
        return final_store

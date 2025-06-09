# groq_utils.py
from llama_index.core import Document, VectorStoreIndex
from groq import Groq
import pandas as pd
import numpy as np

# Keep a global variable to reuse the engine
query_engine = None


def load_product_data(csv_path="product_data.csv"):
    """Loads CSV and converts each product row into a Document object for LlamaIndex"""
    df = pd.read_csv(csv_path)

    # Compute the discount percentage if not already present
    df['discount_percentage'] = np.where(
        (df['compare_at_price'] > 0) & (df['price'] > 0),
        ((df['compare_at_price'] - df['price']) / df['compare_at_price'] * 100),
        0
    )

    # Fill NaN with default values to avoid string conversion errors
    df.fillna({
        "title": "Produit sans titre",
        "vendor": "Inconnu",
        "product_type": "Non défini",
        "price": 0,
        "available": 0,
        "discount_percentage": 0
    }, inplace=True)

    # Create one Document per product row
    documents = [
        Document(
            text=(
                f"Produit: {row['title']}, "
                f"Prix: {row['price']}$, "
                f"Stock: {row['available']} unités, "
                f"Vendeur: {row['vendor']}, "
                f"Type: {row['product_type']}, "
                f"Remise: {row['discount_percentage']:.1f}%"
            )
        )
        for _, row in df.iterrows()
    ]

    return df, documents


def build_query_engine(documents):
    """Builds the LlamaIndex RAG pipeline with Groq"""
    from llama_index.llms.groq import Groq as GroqLLM
    from llama_index.core import Settings

    # Use the proper Groq LLM integration for LlamaIndex
    llm = GroqLLM(
        api_key="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",
        model="llama-3.3-70b-versatile"
    )

    # Configure embedding model to use local/free option
    try:
        # Try using HuggingFace embeddings (free)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except ImportError:
        # Fallback to local embedding
        embed_model = "local"

    # Set both LLM and embedding model
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create index
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


def get_rag_response(prompt, engine=None):
    """Get a RAG-powered response to a user prompt"""
    global query_engine

    if engine is None:
        if query_engine is None:
            _, documents = load_product_data()
            query_engine = build_query_engine(documents)
        engine = query_engine

    response = engine.query(prompt)
    return str(response)


# Alternative approach if the above doesn't work - Custom Groq LLM wrapper
class GroqLlamaIndexLLM:
    """Custom wrapper for Groq to work with LlamaIndex"""

    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.metadata = {"model_name": model}

    def complete(self, prompt: str, **kwargs):
        """Complete method for LlamaIndex compatibility"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.3),
            max_tokens=kwargs.get('max_tokens', 1024),
        )
        return response.choices[0].message.content

    def chat(self, messages, **kwargs):
        """Chat method for LlamaIndex compatibility"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', 0.3),
            max_tokens=kwargs.get('max_tokens', 1024),
        )
        return response.choices[0].message.content

    @property
    def model_name(self):
        return self.model


def build_query_engine_custom(documents):
    """Alternative approach using custom Groq wrapper"""
    from llama_index.core import Settings

    # Create custom Groq LLM instance
    groq_llm = GroqLlamaIndexLLM(
        api_key="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",
        model="llama3-70b-8192"
    )

    # Configure embedding model to avoid OpenAI dependency
    try:
        # Try using HuggingFace embeddings (free)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except ImportError:
        # Fallback to local embedding
        embed_model = "local"

    # Set both LLM and embedding model in Settings
    Settings.llm = groq_llm
    Settings.embed_model = embed_model

    # Create index and query engine
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()
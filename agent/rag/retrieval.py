"""RAG retrieval using TF-IDF"""
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Handle both direct execution and module import
try:
    from .document_loader import DocumentLoader, Document
except ImportError:
    from document_loader import DocumentLoader, Document

class Retriever:
    """TF-IDF based document retriever"""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.documents: List[Document] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams and bigrams
            max_features=1000
        )
        self.tfidf_matrix = None
        self._build_index()
    
    def _build_index(self):
        """Load documents and build TF-IDF index"""
        # Load documents
        loader = DocumentLoader(self.docs_dir)
        self.documents = loader.load_documents()
        
        if not self.documents:
            raise ValueError("No documents loaded!")
        
        # Build TF-IDF matrix
        corpus = [doc.content for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        print(f"✓ Built TF-IDF index with {len(self.documents)} chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Returns:
            List of (Document, score) tuples, sorted by score descending
        """
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])
            results.append((doc, score))
        
        return results
    
    def retrieve_with_threshold(self, query: str, top_k: int = 3, min_score: float = 0.1) -> List[Tuple[Document, float]]:
        """Retrieve documents above a minimum relevance threshold"""
        all_results = self.retrieve(query, top_k)
        return [(doc, score) for doc, score in all_results if score >= min_score]


# BM25 Retriever (optional)
try:
    from rank_bm25 import BM25Okapi
    import re

    class BM25Retriever:
        """BM25 based document retriever (alternative to TF-IDF)"""
        
        def __init__(self, docs_dir: str = "docs"):
            self.docs_dir = docs_dir
            self.documents: List[Document] = []
            self.bm25 = None
            self._build_index()
        
        def _build_index(self):
            """Load documents and build BM25 index"""
            loader = DocumentLoader(self.docs_dir)
            self.documents = loader.load_documents()
            
            if not self.documents:
                raise ValueError("No documents loaded!")
            
            # Tokenize documents
            tokenized_corpus = [self._tokenize(doc.content) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            print(f"✓ Built BM25 index with {len(self.documents)} chunks")
        
        def _tokenize(self, text: str) -> List[str]:
            """Simple tokenization"""
            # Lowercase and split on non-alphanumeric
            tokens = re.findall(r'\w+', text.lower())
            return tokens
        
        def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
            """Retrieve top-k documents using BM25"""
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Build results (normalize scores to 0-1 range)
            max_score = scores.max() if scores.max() > 0 else 1.0
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                score = float(scores[idx] / max_score)
                results.append((doc, score))
            
            return results

except ImportError:
    print("Warning: rank-bm25 not installed. BM25Retriever unavailable.")
    BM25Retriever = None


def get_retriever(method: str = "tfidf", docs_dir: str = "docs"):
    """
    Factory function to get a retriever.
    
    Args:
        method: "tfidf" or "bm25"
        docs_dir: path to documents directory
    
    Returns:
        Retriever instance
    """
    if method == "bm25":
        if BM25Retriever is None:
            raise ImportError("BM25Retriever requires rank-bm25. Install with: pip install rank-bm25")
        return BM25Retriever(docs_dir)
    else:
        return Retriever(docs_dir)


def retrieve_documents(query: str, top_k: int = 3, method: str = "bm25", docs_dir: str = "docs") -> List[Tuple[Document, float]]:
    """
    Retrieve documents for a query.
    
    Returns:
        List of (Document, score) tuples
    """
    retriever = get_retriever(method=method, docs_dir=docs_dir)
    return retriever.retrieve(query, top_k=top_k)


# Test the retriever
if __name__ == "__main__":
    import os
    
    # Determine docs directory based on where script is run from
    if os.path.exists("docs"):
        docs_dir = "docs"
    elif os.path.exists("../../docs"):
        docs_dir = "../../docs"
    else:
        print("Error: Cannot find docs directory")
        exit(1)
    
    retriever = Retriever(docs_dir)
    
    # Test queries
    test_queries = [
        "What is the return policy for beverages?",
        "Summer 1997 marketing campaign dates",
        "How to calculate Average Order Value?",
        "What categories are in the catalog?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve(query, top_k=2)
        
        for doc, score in results:
            print(f"\n[Score: {score:.3f}] {doc.id}")
            print(f"Content: {doc.content[:150]}...")
            
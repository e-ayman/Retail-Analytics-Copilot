"""Test suite for RAG retrieval"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from agent.rag.retrieval import Retriever, BM25Retriever

def test_rag_queries():
    """Test retrieval for eval questions"""
    
    print("Testing TF-IDF Retriever")
    print("="*60)
    
    # Determine docs directory
    if os.path.exists("docs"):
        docs_dir = "docs"
    elif os.path.exists("../../docs"):
        docs_dir = "../../docs"
    else:
        print("Error: Cannot find docs directory")
        return False
    
    retriever = Retriever(docs_dir)
    
    test_cases = [
        {
            "query": "return window days unopened Beverages",
            "expected_source": "product_policy",
            "expected_content": "14 days"
        },
        {
            "query": "Summer Beverages 1997 dates",
            "expected_source": "marketing_calendar",
            "expected_content": "1997-06-01"
        },
        {
            "query": "Average Order Value AOV formula",
            "expected_source": "kpi_definitions",
            "expected_content": "SUM(UnitPrice"
        },
        {
            "query": "Winter Classics 1997 dates December",
            "expected_source": "marketing_calendar",
            "expected_content": "1997-12-01"
        }
    ]
    
    passed = 0
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['query']}")
        results = retriever.retrieve(test['query'], top_k=2)
        
        # Check if expected source in top results
        sources = [doc.source for doc, _ in results]
        content = " ".join([doc.content for doc, _ in results])
        
        source_match = test['expected_source'] in sources
        content_match = test['expected_content'] in content
        
        if source_match and content_match:
            print(f"  ✓ PASS - Found in {results[0][0].id} (score: {results[0][1]:.3f})")
            passed += 1
        else:
            print(f"  ✗ FAIL")
            print(f"    Expected source: {test['expected_source']}, got: {sources}")
            print(f"    Expected content: '{test['expected_content']}'")
            if results:
                print(f"    Top result: {results[0][0].id}: {results[0][0].content[:100]}")
    
    print(f"\n{'='*60}")
    print(f"Passed: {passed}/{len(test_cases)}")
    
    return passed == len(test_cases)

if __name__ == "__main__":
    success = test_rag_queries()
    sys.exit(0 if success else 1)
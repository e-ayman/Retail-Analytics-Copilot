"""Load and chunk markdown documents"""
import os
import re
from typing import List, Dict
from pathlib import Path

class Document:
    """Represents a document chunk"""
    def __init__(self, id: str, content: str, source: str, metadata: dict = None):
        self.id = id
        self.content = content
        self.source = source
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(id={self.id}, source={self.source}, content_preview={self.content[:50]}...)"

class DocumentLoader:
    """Loads and chunks markdown files"""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
    
    def load_documents(self) -> List[Document]:
        """Load all markdown files and chunk them"""
        documents = []
        
        # Get all markdown files
        md_files = sorted(self.docs_dir.glob("*.md"))
        
        for file_path in md_files:
            docs_from_file = self._load_and_chunk_file(file_path)
            documents.extend(docs_from_file)
        
        print(f"✓ Loaded {len(documents)} chunks from {len(md_files)} files")
        return documents
    
    def _load_and_chunk_file(self, file_path: Path) -> List[Document]:
        """Load a single file and split into chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        source_name = file_path.stem  # e.g., "marketing_calendar"
        
        # Split into chunks
        chunks = self._split_into_chunks(content)
        
        documents = []
        for idx, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                doc = Document(
                    id=f"{source_name}::chunk{idx}",
                    content=chunk.strip(),
                    source=source_name,
                    metadata={"file": str(file_path), "chunk_index": idx}
                )
                documents.append(doc)
        
        return documents
    
    def _split_into_chunks(self, content: str) -> List[str]:
        """
        Split content into chunks by:
        1. Markdown headers (##, ###)
        2. Paragraph boundaries
        3. List items grouped together
        
        This creates meaningful, searchable chunks.
        """
        chunks = []
        
        # Split by headers (## or ###) while keeping the header with its content
        # Pattern: Split on headers but include them in the result
        sections = re.split(r'((?:^|\n)#{1,3}\s+.+)', content)
        
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If it's a header, start a new chunk
            if re.match(r'^#{1,3}\s+', section):
                # Save previous chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                # Add content to current chunk
                if current_chunk:
                    current_chunk += "\n" + section
                else:
                    current_chunk = section
            
            # If chunk is getting too large (>300 chars), consider splitting
            if len(current_chunk) > 300:
                # Try to split by double newlines (paragraphs)
                paragraphs = current_chunk.split('\n\n')
                if len(paragraphs) > 1:
                    # Keep first part as a chunk
                    chunks.append(paragraphs[0].strip())
                    # Continue with remaining
                    current_chunk = '\n\n'.join(paragraphs[1:])
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Additional pass: split very large chunks by bullet points
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 400:
                # Try splitting by bullet points
                lines = chunk.split('\n')
                temp_chunk = ""
                for line in lines:
                    if line.strip().startswith('-') and len(temp_chunk) > 100:
                        # Save current and start new
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = line
                    else:
                        temp_chunk += "\n" + line if temp_chunk else line
                
                if temp_chunk:
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks

# Test the loader
if __name__ == "__main__":
    import os
    
    # Determine docs directory
    if os.path.exists("docs"):
        docs_dir = "docs"
    elif os.path.exists("../../docs"):
        docs_dir = "../../docs"
    else:
        print("Error: Cannot find docs directory")
        exit(1)
    
    loader = DocumentLoader(docs_dir)
    documents = loader.load_documents()
    
    print(f"\nTotal chunks: {len(documents)}")
    print("\nAll chunks:")
    for doc in documents:
        print(f"\n{'='*60}")
        print(f"{doc.id}:")
        print(f"  Source: {doc.source}")
        print(f"  Length: {len(doc.content)} chars")
        print(f"  Content:\n{doc.content}")
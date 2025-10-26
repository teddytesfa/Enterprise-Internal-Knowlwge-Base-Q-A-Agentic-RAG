from pathlib import Path
from typing import List, Dict
from llama_index.core import SimpleDirectoryReader, Document

class DocumentConnector:
    """
    Unified document connector for loading various file formats.
    Supports: PDF, Markdown, TXT, DOCX, and more.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.supported_formats = {
            'markdown': ['.md', '.markdown'],
            'pdf': ['.pdf'],
            'text': ['.txt'],
            'docx': ['.docx'],
            'notion': ['.md']  # Notion exports are typically markdown
        }
    
    def get_supported_files(self) -> Dict[str, List[Path]]:
        """Scan directory and categorize files by format."""
        files_by_type = {fmt: [] for fmt in self.supported_formats.keys()}
        
        if not self.data_dir.exists():
            print(f"⚠️  Directory not found: {self.data_dir}")
            return files_by_type
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                for fmt, extensions in self.supported_formats.items():
                    if suffix in extensions:
                        files_by_type[fmt].append(file_path)
                        break
        
        return files_by_type
    
    def load_documents(self, file_extensions: List[str] = None) -> List[Document]:
        """
        Load documents from the data directory.
        
        Args:
            file_extensions: List of file extensions to load (e.g., ['.md', '.pdf'])
                           If None, loads all supported formats.
        
        Returns:
            List of LlamaIndex Document objects
        """
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(self.data_dir),
                required_exts=file_extensions,
                recursive=True,
                filename_as_id=True  # Use filename as document ID
            )
            
            documents = reader.load_data()
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return []
    
    def display_summary(self):
        """Display a summary of available documents."""
        files_by_type = self.get_supported_files()
        
        print("📚 Document Connector Summary")
        print("=" * 60)
        print(f"Data Directory: {self.data_dir}")
        print(f"\nSupported File Types:")
        
        total_files = 0
        for fmt, files in files_by_type.items():
            count = len(files)
            total_files += count
            if count > 0:
                print(f"  • {fmt.capitalize()}: {count} file(s)")
                for file in files:
                    print(f"    - {file.name}")
        
        print(f"\n📊 Total Files: {total_files}")
        print("=" * 60)

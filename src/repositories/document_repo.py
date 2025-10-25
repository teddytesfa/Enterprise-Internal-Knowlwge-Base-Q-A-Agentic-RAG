from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base, Document, Chunk
from typing import Optional, List
import json

class DocumentRepo:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True)

    def upsert_document(self, source_path: str, title: Optional[str], metadata: dict) -> Document:
        session = self.Session()
        try:
            doc = session.query(Document).filter_by(source_path=source_path).one_or_none()
            if doc is None:
                doc = Document(source_path=source_path, title=title, metadata=json.dumps(metadata))
                session.add(doc)
                session.commit()
                session.refresh(doc)
            else:
                doc.title = title or doc.title
                doc.metadata = json.dumps(metadata)
                session.commit()
            return doc
        finally:
            session.close()

    def add_chunks(self, document_id: int, chunks: List[str]):
        session = self.Session()
        try:
            session.query(Chunk).filter_by(document_id=document_id).delete()
            for pos, text in enumerate(chunks):
                c = Chunk(document_id=document_id, text=text, position=pos)
                session.add(c)
            session.commit()
        finally:
            session.close()

    def get_chunk_text(self, document_id: int, position: int) -> Optional[str]:
        session = self.Session()
        try:
            chunk = session.query(Chunk).filter_by(document_id=document_id, position=position).one_or_none()
            return chunk.text if chunk is not None else None
        finally:
            session.close()

    def get_document(self, document_id: int):
        session = self.Session()
        try:
            return session.query(Document).filter_by(id=document_id).one_or_none()
        finally:
            session.close()

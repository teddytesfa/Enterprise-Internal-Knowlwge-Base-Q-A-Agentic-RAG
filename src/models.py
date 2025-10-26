from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy import DateTime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source_path = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=True)
    metadata = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    position = Column(Integer, nullable=False)
    document = relationship("Document", back_populates="chunks")

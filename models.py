from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.types import JSON
from datetime import datetime

DATABASE_URL = "sqlite:///./hiring.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="company")
    jobs = relationship("Job", back_populates="company")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)  # null for superadmin
    username = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="avaliador")  # avaliador | admin | superadmin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="users")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    scorecard = Column(JSON, nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="jobs")
    candidates = relationship("Candidate", back_populates="job", cascade="all, delete-orphan")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    name = Column(String(255), nullable=False)
    transcript = Column(Text, nullable=False)
    evaluation = Column(JSON, nullable=True)
    final_score = Column(Float, nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="candidates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_db():
    """Add new columns to existing tables without destroying data (SQLite compatible)."""
    from sqlalchemy import text, inspect

    with engine.connect() as conn:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        if "jobs" in existing_tables:
            job_cols = {c["name"] for c in inspector.get_columns("jobs")}
            if "company_id" not in job_cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN company_id INTEGER REFERENCES companies(id)"))
            if "is_deleted" not in job_cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"))
            if "deleted_at" not in job_cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN deleted_at DATETIME"))

        if "candidates" in existing_tables:
            cand_cols = {c["name"] for c in inspector.get_columns("candidates")}
            if "is_deleted" not in cand_cols:
                conn.execute(text("ALTER TABLE candidates ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"))
            if "deleted_at" not in cand_cols:
                conn.execute(text("ALTER TABLE candidates ADD COLUMN deleted_at DATETIME"))

        conn.commit()


def create_tables():
    Base.metadata.create_all(bind=engine)
    _migrate_db()

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.types import JSON

load_dotenv()


# ── Database URL helpers ────────────────────────────────────────────────────────

def _normalize_url(raw: str) -> str:
    """
    Normalize a Postgres URL for SQLAlchemy:
    - Fixes postgres:// → postgresql://
    - Strips ?pgbouncer=true (a Supabase/Prisma hint that psycopg2 does not understand)
    """
    url = raw.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    url = url.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
    return url


_raw_url = os.getenv("DATABASE_URL")
if not _raw_url or not _raw_url.strip():
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Add your Supabase connection string to the .env file or deployment environment."
    )

# ── Engines ────────────────────────────────────────────────────────────────────
#
# DATABASE_URL  → Supabase PgBouncer pooler (port 6543). Used for all app queries.
#                 NullPool is used so serverless invocations don't compete over
#                 a shared pool — PgBouncer already handles pooling externally.
#
# DIRECT_URL    → Direct Postgres connection (port 5432). Required for DDL
#                 (CREATE TABLE / ALTER TABLE) which PgBouncer transaction mode
#                 does not support reliably. Falls back to DATABASE_URL if unset.

DATABASE_URL = _normalize_url(_raw_url)
engine = create_engine(DATABASE_URL, poolclass=NullPool)

_raw_direct = os.getenv("DIRECT_URL") or _raw_url
_direct_engine = create_engine(_normalize_url(_raw_direct), poolclass=NullPool)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ─────────────────────────────────────────────────────────────────────

class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    # status: pending | active | blocked
    status = Column(String(20), nullable=False, default="active")
    # kept for backward compat — status is canonical going forward
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="company")
    jobs = relationship("Job", back_populates="company")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)  # null for superadmin
    username = Column(String(255), nullable=True, unique=True)  # deprecated — use email
    email = Column(String(255), nullable=True, unique=True, index=True)
    email_verified = Column(Boolean, default=True)
    must_change_password = Column(Boolean, default=False)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="avaliador")  # avaliador | admin | superadmin
    # status: pending | active | blocked
    status = Column(String(20), nullable=False, default="active")
    # kept for backward compat — status is canonical going forward
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="users")
    created_jobs = relationship("Job", back_populates="created_by", foreign_keys="Job.created_by_user_id")
    responsible_jobs = relationship("Job", back_populates="responsavel", foreign_keys="Job.responsavel_user_id")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    responsavel_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    scorecard = Column(JSON, nullable=True)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="jobs")
    created_by = relationship("User", back_populates="created_jobs", foreign_keys="Job.created_by_user_id")
    responsavel = relationship("User", back_populates="responsible_jobs", foreign_keys="Job.responsavel_user_id")
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


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    prompt_text = Column(Text, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_db():
    """
    Idempotently add new columns to existing tables.
    Runs via the direct connection (DIRECT_URL) so DDL works correctly
    even when the app engine is behind PgBouncer.
    """
    from sqlalchemy import inspect, text

    with _direct_engine.connect() as conn:
        inspector = inspect(_direct_engine)
        existing_tables = inspector.get_table_names()

        def _col_names(table: str) -> set:
            return {c["name"] for c in inspector.get_columns(table)}

        # ── jobs ──────────────────────────────────────────────────────────────
        if "jobs" in existing_tables:
            cols = _col_names("jobs")
            if "company_id" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN company_id INTEGER REFERENCES companies(id)"))
            if "is_deleted" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT FALSE"))
            if "deleted_at" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN deleted_at TIMESTAMP"))
            if "created_by_user_id" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN created_by_user_id INTEGER REFERENCES users(id)"))
            if "responsavel_user_id" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN responsavel_user_id INTEGER REFERENCES users(id)"))

            # ── Always-run backfills (idempotent — only fill NULLs) ─────────
            # 1. Jobs linked to a company: assign oldest active admin (or user) of that company.
            conn.execute(text("""
                UPDATE jobs j
                SET created_by_user_id = (
                    SELECT u.id FROM users u
                    WHERE u.company_id = j.company_id
                      AND u.status != 'blocked'
                    ORDER BY
                        CASE WHEN u.role = 'admin' THEN 0 ELSE 1 END,
                        u.created_at ASC
                    LIMIT 1
                )
                WHERE j.created_by_user_id IS NULL
                  AND j.company_id IS NOT NULL
            """))
            # 2. Jobs with no company (superadmin-created): assign the oldest superadmin.
            conn.execute(text("""
                UPDATE jobs j
                SET created_by_user_id = (
                    SELECT u.id FROM users u
                    WHERE u.role = 'superadmin'
                      AND u.status != 'blocked'
                    ORDER BY u.created_at ASC
                    LIMIT 1
                )
                WHERE j.created_by_user_id IS NULL
                  AND j.company_id IS NULL
            """))
            # 3. Responsável defaults to the creator when not explicitly set.
            conn.execute(text("""
                UPDATE jobs SET responsavel_user_id = created_by_user_id
                WHERE responsavel_user_id IS NULL AND created_by_user_id IS NOT NULL
            """))
            # 4. Catch-all: any job still without a responsável gets the oldest superadmin.
            conn.execute(text("""
                UPDATE jobs SET responsavel_user_id = (
                    SELECT u.id FROM users u
                    WHERE u.role = 'superadmin'
                      AND u.status != 'blocked'
                    ORDER BY u.created_at ASC
                    LIMIT 1
                )
                WHERE responsavel_user_id IS NULL
            """))

        # ── candidates ────────────────────────────────────────────────────────
        if "candidates" in existing_tables:
            cols = _col_names("candidates")
            if "is_deleted" not in cols:
                conn.execute(text("ALTER TABLE candidates ADD COLUMN is_deleted BOOLEAN NOT NULL DEFAULT FALSE"))
            if "deleted_at" not in cols:
                conn.execute(text("ALTER TABLE candidates ADD COLUMN deleted_at TIMESTAMP"))

        # ── companies ─────────────────────────────────────────────────────────
        if "companies" in existing_tables:
            cols = _col_names("companies")
            if "status" not in cols:
                conn.execute(text("ALTER TABLE companies ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active'"))

        # ── users ─────────────────────────────────────────────────────────────
        if "users" in existing_tables:
            cols = _col_names("users")
            if "status" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active'"))
                conn.execute(text(
                    "UPDATE users SET status = CASE WHEN is_active = TRUE THEN 'active' ELSE 'blocked' END"
                ))
            if "email" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR(255)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email) WHERE email IS NOT NULL"))
            if "email_verified" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN NOT NULL DEFAULT TRUE"))
            if "must_change_password" not in cols:
                conn.execute(text("ALTER TABLE users ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT FALSE"))
            # Make username nullable if it still has a NOT NULL constraint
            try:
                conn.execute(text("ALTER TABLE users ALTER COLUMN username DROP NOT NULL"))
            except Exception:
                pass  # already nullable

        conn.commit()


def create_tables():
    Base.metadata.create_all(bind=_direct_engine)
    _migrate_db()

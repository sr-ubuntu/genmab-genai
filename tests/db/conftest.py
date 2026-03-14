"""Shared fixtures for the db subpackage tests."""
import pytest
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

from clinical_summarizer.db.models import Base
from clinical_summarizer.db.seed import seed_database


@pytest.fixture
def in_memory_engine() -> Engine:
    """SQLite in-memory engine with all tables created.

    Yields:
        A fresh Engine instance pointing to an in-memory SQLite database.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def seeded_session(in_memory_engine: Engine) -> Session:
    """A Session with seed data inserted, rolled back after each test.

    Uses the in_memory_engine and seeds it with mock patient and record data.

    Args:
        in_memory_engine: Fixture that provides a fresh in-memory Engine.

    Yields:
        An open Session with seeded data. Automatically rolled back after test.
    """
    SessionLocal = sessionmaker(bind=in_memory_engine)
    session = SessionLocal()
    try:
        seed_database(session)
        yield session
    finally:
        session.rollback()
        session.close()

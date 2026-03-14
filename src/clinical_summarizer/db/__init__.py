"""Database layer for the clinical summarizer.

Exports the SQLAlchemy engine and session factory for use throughout
the application.
"""
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker


__all__ = ["get_engine", "get_session_factory"]


def get_engine(db_url: str = "sqlite:///clinical.db") -> Engine:
    """Create and return a SQLAlchemy Engine for the given database URL.

    Args:
        db_url: SQLAlchemy connection string. Defaults to a local SQLite file.

    Returns:
        A configured Engine instance.
    """
    from sqlalchemy import create_engine

    return create_engine(db_url, echo=False)


def get_session_factory(engine: Engine):
    """Return a sessionmaker bound to the provided engine.

    Args:
        engine: A bound SQLAlchemy Engine instance.

    Returns:
        A sessionmaker callable that produces new Session instances.
    """
    return sessionmaker(bind=engine)

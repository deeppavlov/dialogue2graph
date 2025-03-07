import os
import logging
import uuid
from dotenv import load_dotenv

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyCache, InMemoryCache

# from langchain_core.load.load import loads
# from langchain_core.load.dump import dumps
# from langchain_core.outputs import Generation

from sqlalchemy import create_engine

# from sqlalchemy.sql import func
# from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
load_dotenv()


# class PromptCacheSchema(Base):  # type: ignore
#     """SQLite table for full LLM Cache (all generations)."""

#     __tablename__ = "prompt_cache"

#     idx = Column(Integer, primary_key=True)

#     prompt = Column(String, primary_key=True)

#     response = Column(String)

#     llm = Column(String, primary_key=True)

#     created_at = Column(DateTime(timezone=True), server_default=func.now())


# class PromptCache(SQLAlchemyCache):

#     def lookup(self, prompt: str, llm_string: str):
#         """Look up based on prompt and llm_string."""

#         stmt = (
#             select(self.cache_schema.response)
#             .where(self.cache_schema.prompt == prompt)  # type: ignore
#             .where(self.cache_schema.llm == llm_string)
#             .order_by(self.cache_schema.idx)
#         )

#         with Session(self.engine) as session:

#             rows = session.execute(stmt).fetchall()

#             if rows:

#                 try:

#                     return [loads(row[0]) for row in rows]

#                 except Exception:

#                     logger.warning(
#                         "Retrieving a cache value that could not be deserialized "
#                         "properly. This is likely due to the cache being in an "
#                         "older format. Please recreate your cache to avoid this "
#                         "error."
#                     )

#                     # In a previous life we stored the raw text directly

#                     # in the table, so assume it's in that format.

#                     return [Generation(text=row[0]) for row in rows]

#         return None

#     def update(self, prompt: str, llm_string: str, return_val) -> None:
#         """Update based on prompt and llm_string."""

#         items = [self.cache_schema(prompt=prompt, llm=llm_string, response=dumps(gen), idx=i) for i, gen in enumerate(return_val)]

#         with Session(self.engine) as session, session.begin():

#             for item in items:

#                 session.merge(item)


def add_uuid_to_prompt(prompt: str, seed: int = None) -> str:
    """Add a UUID to the beginning of a prompt."""
    if seed is not None:
        # Create a UUID using the seed
        random_uuid = uuid.UUID(int=seed)
    else:
        random_uuid = uuid.uuid4()
    return f"UUID: {random_uuid}\n\n{prompt}"


def setup_cache(use_in_memory: bool = False):
    """Set up the LLM cache.

    Args:
        use_in_memory: If True, use InMemoryCache instead of SQLAlchemyCache

    Returns:
        The configured cache object or None if setup fails
    """
    try:
        if use_in_memory:
            cache = InMemoryCache()
            set_llm_cache(cache)
            logger.info("Using in-memory cache")
            return cache

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.warning("DATABASE_URL not found in environment, falling back to in-memory cache")
            return setup_cache(use_in_memory=True)

        engine = create_engine(database_url)
        cache = SQLAlchemyCache(engine)
        set_llm_cache(cache)
        logger.info("Successfully set up SQLAlchemy cache")
        return cache

    except Exception as e:
        logger.warning(f"Error setting up cache: {e}. Falling back to in-memory cache")
        return setup_cache(use_in_memory=True)

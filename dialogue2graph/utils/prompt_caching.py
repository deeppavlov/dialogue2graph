import os
import logging
from dotenv import load_dotenv

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLAlchemyCache, Base
from langchain_core.load.load import loads
from langchain_core.load.dump import dumps
from langchain_core.outputs import Generation

from sqlalchemy import Column, Integer, String, create_engine, select, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
load_dotenv()


class PromptCacheSchema(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "prompt_cache"

    id = Column(Integer, primary_key=True)

    prompt = Column(String, primary_key=True)

    response = Column(String)

    llm = Column(String, primary_key=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PromptCache(SQLAlchemyCache):

    def lookup(self, prompt: str, llm_string: str):
        """Look up based on prompt and llm_string."""

        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt == prompt)  # type: ignore
            .where(self.cache_schema.llm == llm_string)
            .order_by(self.cache_schema.id)
        )

        with Session(self.engine) as session:

            rows = session.execute(stmt).fetchall()

            if rows:

                try:

                    return [loads(row[0]) for row in rows]

                except Exception:

                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )

                    # In a previous life we stored the raw text directly

                    # in the table, so assume it's in that format.

                    return [Generation(text=row[0]) for row in rows]

        return None

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        """Update based on prompt and llm_string."""

        items = [self.cache_schema(prompt=prompt, llm=llm_string, response=dumps(gen), id=i) for i, gen in enumerate(return_val)]

        with Session(self.engine) as session, session.begin():

            for item in items:

                session.merge(item)


def setup_cache():
    """Set up the LLM cache."""
    try:
        engine = create_engine(os.getenv("DATABASE_URL"))
        cache = PromptCache(engine, PromptCacheSchema)
        set_llm_cache(cache)
    except Exception as e:
        logger.error(f"Error setting up cache: {e}")
        return None

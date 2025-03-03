-- Create the database if it doesn't exist
-- Note: This isn't needed if POSTGRES_DB is set in docker-compose.yml
-- CREATE DATABASE prompt_cache_db;

-- Connect to the database
\c prompt_cache_db

-- Create the table
CREATE TABLE IF NOT EXISTS prompt_cache (
    id SERIAL PRIMARY KEY,  -- Changed to SERIAL for auto-incrementing
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    llm TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user with environment variable
CREATE USER prompt_cache_user WITH PASSWORD '${PROMPT_CACHE_PASSWORD}';

-- Grant permissions
GRANT CONNECT ON DATABASE prompt_cache_db TO prompt_cache_user;
GRANT USAGE ON SCHEMA public TO prompt_cache_user;
GRANT SELECT, INSERT ON prompt_cache TO prompt_cache_user;
GRANT USAGE, SELECT ON SEQUENCE prompt_cache_id_seq TO prompt_cache_user;
CREATE TABLE IF NOT EXISTS prompt_cache (
    idx SERIAL PRIMARY KEY,  -- Changed from INT to SERIAL
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
GRANT USAGE, SELECT ON SEQUENCE prompt_cache_idx_seq TO prompt_cache_user;
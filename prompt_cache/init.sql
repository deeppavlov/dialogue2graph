CREATE TABLE prompt_cache (
    id INTEGER PRIMARY KEY,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    llm TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
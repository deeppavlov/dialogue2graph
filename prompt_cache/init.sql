-- Create a dedicated database for LangChain cache
CREATE DATABASE langchain_cache;

-- Create a role with limited privileges (default password will be overridden)
DO $$
BEGIN
  EXECUTE format('CREATE ROLE langchain_user WITH LOGIN PASSWORD %L', 
                current_setting('app.prompt_cache_password'));
EXCEPTION WHEN undefined_object THEN
  -- Fallback if the variable isn't set
  CREATE ROLE langchain_user WITH LOGIN PASSWORD 'langchain_pass';
END
$$;

-- Rest of your SQL remains the same...
GRANT CONNECT ON DATABASE langchain_cache TO langchain_user;
\c langchain_cache
GRANT USAGE ON SCHEMA public TO langchain_user;
GRANT CREATE ON SCHEMA public TO langchain_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO langchain_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO langchain_user;
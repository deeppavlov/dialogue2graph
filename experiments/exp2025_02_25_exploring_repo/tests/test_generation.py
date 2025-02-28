from dotenv import load_dotenv


def test_env_file():
    """Test an env file existence"""
    assert load_dotenv()

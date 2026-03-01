from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):

    HF_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    JINA_API: Optional[str] = None

    model_config = SettingsConfigDict(env_file=Path(__file__).parent.parent / ".env", extra="ignore")

CONFIG = Settings()
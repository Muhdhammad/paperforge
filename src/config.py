from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):

    HF_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

CONFIG = Settings()
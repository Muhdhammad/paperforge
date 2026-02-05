from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    HF_TOKEN: str
    OPENAI_API_KEY: str
    GROQ_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

CONFIG = Settings()
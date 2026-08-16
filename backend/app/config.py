import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "hh-goa-voice-rag"
    dataset_name: str = os.getenv("DATASET_NAME", "ai4bharat/MSMARCO-XI")
    hf_token: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ewcl_bundle_dir: Path = Path(__file__).resolve().parents[1] / "backend_bundle"

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()



from pydantic_settings import BaseSettings
from pydantic import ValidationError  # noqa: F401 — re-exported for convenience


class Settings(BaseSettings):
    langsmith_api_key: str
    langsmith_project: str = "socratico_test"

    # Tracing configs
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "qa-analyst-traces"

    # Gemini / Google AI key.  We read both common env var names so the user
    # doesn't need to duplicate their key.
    gemini_api_key: str = ""
    google_api_key: str = ""


    @property
    def effective_gemini_key(self) -> str:
        """Returns whichever key is set (GEMINI_API_KEY takes priority)."""
        return self.gemini_api_key or self.google_api_key

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()

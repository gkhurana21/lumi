from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    providers: str = "fake"  # "fake" | "cloud"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    vision_model: str = "claude-haiku-4-5-20251001"
    stt_model: str = "gpt-4o-mini-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    scene_interval_s: float = 4.0
    urdf_path: str = "robot/dummy_lamp_5dof.urdf"
    greet_cooldown_s: float = 45.0
    memory: str = "chroma"          # "chroma" | "inmem" (no deps, keyword match)
    memory_path: str = "data/chroma"
    attention: str = "mediapipe"    # "mediapipe" | "none" (headless tests)
    sim: bool = True
    trace_dir: str = "out/traces"  # per-session JSONL of attention readings + transitions (local); "" = off


settings = Settings()

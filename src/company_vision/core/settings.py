
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppConfig(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    environment: Literal["development", "test", "production"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class CameraConfig(StrictModel):
    camera_id: str = Field(min_length=1)
    source_type: Literal["video", "webcam", "rtsp"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    target_fps: int = Field(gt=0, le=120)
    reconnect_seconds: int = Field(ge=1, le=300)


class ModelsConfig(StrictModel):
    device: str = Field(min_length=1)
    pose_model: str = Field(min_length=1)
    detection_model: str = Field(min_length=1)
    pose_confidence: float = Field(ge=0.0, le=1.0)
    detection_confidence: float = Field(ge=0.0, le=1.0)


class TrackingConfig(StrictModel):
    tracker: str = Field(min_length=1)
    min_confidence: float = Field(ge=0.0, le=1.0)


class PrivacyConfig(StrictModel):
    save_raw_frames: bool = False
    save_face_crops: bool = False
    save_observations: bool = False
    retention_days: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_retention(self):
        if self.save_observations and self.retention_days == 0:
            raise ValueError(
                "retention_days must be positive "
                "when saving observations"
            )
        return self


class Settings(StrictModel):
    app: AppConfig
    camera: CameraConfig
    models: ModelsConfig
    tracking: TrackingConfig
    privacy: PrivacyConfig


def load_settings(
    path: str | Path = "configs/app.yaml",
) -> Settings:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r", encoding="utf-8-sig"
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise TypeError(
            "Configuration root must be a YAML mapping"
        )

    return Settings.model_validate(data)

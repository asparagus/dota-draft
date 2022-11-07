from attrs import define
from typing import Optional
import yaml


def yaml_load(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


@define
class Gcs:
    """Configuration used for Google Cloud Storage."""
    bucket: str
    project: str


@define
class Wandb:
    """Configuration used for Wandb."""
    project: str


@define
class OpenDota:
    """Configuration used for OpenDota."""
    api_key: Optional[str]


GCS = Gcs(**yaml_load('draft/configs/gcs.yaml'))
OPENDOTA = OpenDota(**yaml_load('draft/configs/opendota.yaml'))
WANDB = Wandb(**yaml_load('draft/configs/wandb.yaml'))

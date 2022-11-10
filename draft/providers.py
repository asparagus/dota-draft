"""Module with parsed configurations to the service providers used in this project."""
from attrs import define
from typing import Optional
import yaml


def yaml_load(path: str):
    """Load a yaml file and return the parsed content.

    Args:
        path: The path to the yaml file

    Returns:
        The parsed yaml file as a dict
    """
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

"""Module with parsed configurations to the service providers used in this project."""
from attrs import define
from typing import Optional
import os
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
class Gar:
    """Configuration used for Google Artifact Registry."""
    location: str
    project: str
    repository: str


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


PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))

GAR = Gar(**yaml_load(os.path.join(PACKAGE_ROOT, 'configs/gar.yaml')))
GCS = Gcs(**yaml_load(os.path.join(PACKAGE_ROOT, 'configs/gcs.yaml')))
OPENDOTA = OpenDota(**yaml_load(os.path.join(PACKAGE_ROOT, 'configs/opendota.yaml')))
WANDB = Wandb(**yaml_load(os.path.join(PACKAGE_ROOT, 'configs/wandb.yaml')))

import os
from enum import Enum


class ENVS(Enum):
    """Allowed types of environments
    """
    Dev = "dev"
    Prod = "prod"
    Dev_Prod = "dev-prod"


ENV = os.getenv('ENV', ENVS.Dev_Prod)

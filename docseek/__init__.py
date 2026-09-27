from pathlib import Path

from dotenv import load_dotenv

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SERVICE_ENV = _PACKAGE_ROOT.parent / '.env'

load_dotenv(_SERVICE_ENV, override=False)
load_dotenv(override=False)

from .models import ANode, AgentAction, FullElement
from .scraper import fetch_and_build_tree

__all__ = [
    'ANode',
    'AgentAction',
    'FullElement',
    'fetch_and_build_tree',
]

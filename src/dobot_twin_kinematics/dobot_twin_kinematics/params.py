"""Loads config/limits.yaml (installed share dir, or source tree) into a ready-to-use Magician model."""
import os
from typing import Optional

import yaml

from .kinematics import Geometry, Limits, Magician


def default_limits_file() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('dobot_twin_kinematics'), 'config', 'limits.yaml')
        if os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001  (not installed: source-tree fallback)
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'limits.yaml')


def load_config(path: Optional[str] = None, profile: Optional[str] = None) -> dict:
    with open(path or default_limits_file()) as f:
        cfg = yaml.safe_load(f)
    name = profile or cfg['active_profile']
    if name not in cfg['profiles']:
        raise ValueError(f"unknown limits profile '{name}', available: {list(cfg['profiles'])}")
    cfg['profile_name'] = name
    return cfg


def load_magician(path: Optional[str] = None, profile: Optional[str] = None) -> Magician:
    cfg = load_config(path, profile)
    geo = Geometry(tcp_dz=float(cfg.get('tcp_dz', Geometry.tcp_dz)))
    return Magician(geo, Limits.from_deg_dict(cfg['profiles'][cfg['profile_name']]))

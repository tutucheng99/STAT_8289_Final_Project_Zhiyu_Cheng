"""
Run Metadata Logger
====================
Automatically records experiment metadata for reproducibility.

Usage:
    from src.meta_logger import setup_run, save_run_metadata

    # At the start of a script
    run_id, output_dir = setup_run(cfg)

    # Metadata is automatically saved to output_dir/meta/
"""
import os
import json
import platform
import subprocess
from datetime import datetime
from typing import Tuple, Optional

try:
    from omegaconf import OmegaConf, DictConfig
except ImportError:
    OmegaConf = None
    DictConfig = dict


def get_git_hash() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return "unknown"


def get_git_diff_status() -> str:
    """Check if there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "dirty" if result.stdout.strip() else "clean"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return "unknown"


def get_pip_freeze() -> str:
    """Get pip freeze output."""
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return "unavailable"


def generate_run_id(name: str = "run") -> str:
    """
    Generate a unique run ID.

    Format: YYYYMMDD_HHMMSS_<name>_<git_hash>
    Example: 20251215_143022_fda_abc1234
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()
    return f"{timestamp}_{name}_{git_hash}"


def setup_run(cfg, project_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Set up a new run with automatic metadata logging.

    Args:
        cfg: OmegaConf config object
        project_root: Project root directory (defaults to cwd)

    Returns:
        Tuple of (run_id, output_dir)
    """
    if project_root is None:
        project_root = os.getcwd()

    # Generate run ID
    run_name = cfg.run.name if hasattr(cfg.run, 'name') else "run"
    run_id = generate_run_id(run_name)

    # Create output directory
    output_base = cfg.run.output_dir if hasattr(cfg.run, 'output_dir') else "results"
    output_dir = os.path.join(project_root, output_base, run_id)

    # Check overwrite policy
    if os.path.exists(output_dir):
        if hasattr(cfg.run, 'overwrite') and not cfg.run.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Set run.overwrite=true to allow overwriting."
            )

    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    save_run_metadata(cfg, output_dir)

    return run_id, output_dir


def save_run_metadata(cfg, output_dir: str) -> None:
    """
    Save run metadata to output_dir/meta/.

    Saves:
        - config.yaml: Full configuration
        - git.txt: Git commit hash and status
        - freeze.txt: pip freeze output (if enabled)
        - run.json: Run information (timestamp, platform, etc.)
    """
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # 1. Save configuration
    config_path = os.path.join(meta_dir, "config.yaml")
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        OmegaConf.save(cfg, config_path)
    else:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(dict(cfg) if hasattr(cfg, '__iter__') else {}, f)

    # 2. Save git info
    git_path = os.path.join(meta_dir, "git.txt")
    git_hash = get_git_hash()
    git_status = get_git_diff_status()
    with open(git_path, 'w') as f:
        f.write(f"commit: {git_hash}\n")
        f.write(f"status: {git_status}\n")

    # 3. Save pip freeze (if enabled in config)
    save_freeze = True
    if hasattr(cfg, 'repro') and hasattr(cfg.repro, 'save_pip_freeze'):
        save_freeze = cfg.repro.save_pip_freeze

    if save_freeze:
        freeze_path = os.path.join(meta_dir, "freeze.txt")
        with open(freeze_path, 'w') as f:
            f.write(get_pip_freeze())

    # 4. Save run information
    run_info = {
        "run_id": os.path.basename(output_dir),
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
    }

    # Add config-derived info
    if hasattr(cfg, 'repro') and hasattr(cfg.repro, 'seed'):
        run_info["seed"] = cfg.repro.seed
    if hasattr(cfg, 'compute') and hasattr(cfg.compute, 'platform'):
        run_info["compute_platform"] = cfg.compute.platform

    run_path = os.path.join(meta_dir, "run.json")
    with open(run_path, 'w') as f:
        json.dump(run_info, f, indent=2)

    print(f"Metadata saved to: {meta_dir}")


def load_run_metadata(output_dir: str) -> dict:
    """Load run metadata from a previous run."""
    meta_dir = os.path.join(output_dir, "meta")
    run_path = os.path.join(meta_dir, "run.json")

    if os.path.exists(run_path):
        with open(run_path, 'r') as f:
            return json.load(f)
    return {}


# Convenience function for scripts
def init_experiment(config_path: str, overrides: list = None) -> Tuple[DictConfig, str, str]:
    """
    Initialize an experiment with config loading and metadata setup.

    Args:
        config_path: Path to config YAML file
        overrides: List of config overrides (e.g., ["repro.seed=123"])

    Returns:
        Tuple of (cfg, run_id, output_dir)
    """
    if OmegaConf is None:
        raise ImportError("OmegaConf is required. Install via: pip install omegaconf")

    # Load config
    cfg = OmegaConf.load(config_path)

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Setup run
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    run_id, output_dir = setup_run(cfg, project_root)

    return cfg, run_id, output_dir

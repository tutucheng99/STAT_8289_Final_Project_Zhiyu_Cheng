"""
Data Preparation Script for Bridge Bidding Interpretability Project

This script:
1. Validates DDS data files (existence, format, shape)
2. Samples 100K evaluation set from 500K DDS data (fixed seed for reproducibility)
3. Validates OpenSpiel SL data
4. Generates data_manifest.json with file metadata

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/default_config.yaml
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def compute_file_hash(path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    hash_obj = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def validate_dds_file(path: str) -> dict:
    """Validate a DDS numpy file and return metadata."""
    if not os.path.exists(path):
        return {"path": path, "exists": False, "error": "File not found"}

    try:
        data = np.load(path, mmap_mode="r")
        return {
            "path": str(path),
            "exists": True,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "size_bytes": os.path.getsize(path),
            "min": int(data.min()),
            "max": int(data.max()),
            "hash_sha256": compute_file_hash(path),
        }
    except Exception as e:
        return {"path": path, "exists": True, "error": str(e)}


def validate_openspiel_file(path: str, n_sample_lines: int = 100) -> dict:
    """
    Validate an OpenSpiel text file and return metadata.

    OpenSpiel bridge SL format (from WBridge5 data):
    - Each line contains space-separated integers
    - First 52 values: card positions (deck permutation)
    - Remaining values: bidding sequence (actions in [52, 91] range + pass/double/redouble)
    - Last value: the action label

    NOTE: This format is DIFFERENT from PGX's 480-dim binary observation.
    The brl SL training converts this to the neural network input format.
    We use pre-trained models, so we only verify the files exist and are parseable.
    """
    if not os.path.exists(path):
        return {"path": path, "exists": False, "error": "File not found"}

    try:
        with open(path, "r") as f:
            # Count lines without loading entire file into memory
            line_count = sum(1 for _ in f)

        # Get first line for format check
        with open(path, "r") as f:
            first_line = f.readline().strip()

        # Sample and validate lines
        validation_results = {
            "n_sampled": 0,
            "n_parseable": 0,
            "n_correct_features": 0,
            "n_valid_action": 0,
            "feature_dims_seen": set(),
            "action_range_seen": [float('inf'), float('-inf')],
            "errors": [],
        }

        # Sample evenly distributed lines
        sample_indices = set(
            int(i * line_count / n_sample_lines)
            for i in range(n_sample_lines)
        )

        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i not in sample_indices:
                    continue

                validation_results["n_sampled"] += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split()
                    if len(parts) < 2:
                        validation_results["errors"].append(f"Line {i}: too few parts")
                        continue

                    validation_results["n_parseable"] += 1

                    # Last value is action, rest are features
                    features = [float(x) for x in parts[:-1]]
                    action = int(parts[-1])

                    # Check feature dimension
                    validation_results["feature_dims_seen"].add(len(features))
                    if len(features) == 480:
                        validation_results["n_correct_features"] += 1

                    # Check action range
                    validation_results["action_range_seen"][0] = min(
                        validation_results["action_range_seen"][0], action
                    )
                    validation_results["action_range_seen"][1] = max(
                        validation_results["action_range_seen"][1], action
                    )
                    if 0 <= action <= 37:
                        validation_results["n_valid_action"] += 1
                    else:
                        validation_results["errors"].append(
                            f"Line {i}: action {action} out of range [0,37]"
                        )

                except Exception as e:
                    validation_results["errors"].append(f"Line {i}: {str(e)}")

        # Convert set to list for JSON serialization
        validation_results["feature_dims_seen"] = list(validation_results["feature_dims_seen"])

        return {
            "path": str(path),
            "exists": True,
            "line_count": line_count,
            "size_bytes": os.path.getsize(path),
            "first_line_preview": first_line[:100] + "..." if len(first_line) > 100 else first_line,
            "hash_sha256": compute_file_hash(path),
            "format_validation": {
                "n_sampled": validation_results["n_sampled"],
                "n_parseable": validation_results["n_parseable"],
                "n_correct_features_480": validation_results["n_correct_features"],
                "n_valid_action_0_37": validation_results["n_valid_action"],
                "feature_dims_seen": validation_results["feature_dims_seen"],
                "action_range": validation_results["action_range_seen"],
                "sample_errors": validation_results["errors"][:5],  # Limit error output
            },
        }
    except Exception as e:
        return {"path": path, "exists": True, "error": str(e)}


def prepare_dds_eval(
    source_path: str,
    output_path: str,
    n_samples: int = 100000,
    seed: int = 42,
) -> dict:
    """
    Sample evaluation set from DDS data with fixed seed.

    DDS data format: [2, N, 4] where:
    - dim 0: [board_deals, dds_results] (2 arrays)
    - dim 1: N boards
    - dim 2: 4 values per board

    We sample along axis 1 (board dimension).

    Args:
        source_path: Path to source DDS file (e.g., 500K)
        output_path: Path to output evaluation file
        n_samples: Number of samples to extract
        seed: Random seed for reproducibility

    Returns:
        Metadata dictionary
    """
    print(f"Loading source data from {source_path}...")
    data = np.load(source_path, mmap_mode="r")

    # DDS data is [2, N, 4], sample along axis 1
    n_boards = data.shape[1]

    if n_boards < n_samples:
        raise ValueError(
            f"Source data has {n_boards} boards, but {n_samples} requested"
        )

    print(f"Sampling {n_samples} from {n_boards} boards (seed={seed})...")
    np.random.seed(seed)
    indices = np.random.choice(n_boards, size=n_samples, replace=False)
    indices.sort()  # Sort for efficient mmap access

    print("Extracting samples...")
    # Sample along axis 1: [2, n_samples, 4]
    eval_data = data[:, indices, :].copy()  # Copy to avoid mmap issues

    print(f"Saving to {output_path}...")
    np.save(output_path, eval_data)

    # Save indices for reproducibility verification
    indices_path = output_path.replace(".npy", "_indices.npy")
    np.save(indices_path, indices)
    print(f"  Indices saved to: {indices_path}")

    return {
        "source": str(source_path),
        "derived_from": Path(source_path).name,
        "output": str(output_path),
        "n_samples": n_samples,
        "seed": seed,
        "source_boards": n_boards,
        "output_shape": list(eval_data.shape),
        "dtype": str(eval_data.dtype),
        "hash_sha256": compute_file_hash(output_path),
        "indices_path": indices_path,
        "indices_sha256": compute_file_hash(indices_path),
    }


def generate_data_manifest(project_root: str, eval_info: dict = None) -> dict:
    """Generate comprehensive data manifest with lineage tracking."""
    raw_dir = Path(project_root) / "data" / "raw"
    dds_dir = raw_dir / "dds_results"
    openspiel_dir = raw_dir / "openspiel_bridge"

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(project_root),
        "dds_data": {},
        "openspiel_data": {},
        "derived_data": {},
    }

    # Validate DDS files
    dds_files = [
        "dds_results_10M.npy",
        "dds_results_2.5M.npy",
        "dds_results_500K.npy",
    ]

    for fname in dds_files:
        fpath = dds_dir / fname
        manifest["dds_data"][fname] = validate_dds_file(str(fpath))

    # Check for derived eval file with lineage info
    eval_file = dds_dir / "dds_results_100K_eval.npy"
    indices_file = dds_dir / "dds_results_100K_eval_indices.npy"

    if eval_file.exists():
        eval_meta = validate_dds_file(str(eval_file))

        # Add lineage information
        eval_meta["lineage"] = {
            "derived_from": "dds_results_500K.npy",
            "derivation_method": "random_sampling_without_replacement",
            "seed": 42,
            "n_samples": 100000,
        }

        # Add indices file info if exists
        if indices_file.exists():
            eval_meta["lineage"]["indices_file"] = str(indices_file)
            eval_meta["lineage"]["indices_sha256"] = compute_file_hash(str(indices_file))

        # Include eval_info if provided from current run
        if eval_info:
            eval_meta["lineage"].update({
                "source_boards": eval_info.get("source_boards"),
                "indices_sha256": eval_info.get("indices_sha256"),
            })

        manifest["derived_data"]["dds_results_100K_eval.npy"] = eval_meta

    # Validate OpenSpiel files with format checking
    openspiel_files = ["train.txt", "test.txt"]
    for fname in openspiel_files:
        fpath = openspiel_dir / fname
        manifest["openspiel_data"][fname] = validate_openspiel_file(str(fpath))

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare data for bridge bidding project")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (auto-detected if not specified)",
    )
    parser.add_argument(
        "--skip-eval-sampling",
        action="store_true",
        help="Skip 100K evaluation set sampling",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100000,
        help="Number of evaluation samples (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    # Auto-detect project root
    if args.project_root is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
    else:
        project_root = Path(args.project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)

    raw_dir = project_root / "data" / "raw"
    dds_dir = raw_dir / "dds_results"

    # Step 1: Validate existing DDS files
    print("\n[1/4] Validating DDS files...")
    dds_files = {
        "10M": dds_dir / "dds_results_10M.npy",
        "2.5M": dds_dir / "dds_results_2.5M.npy",
        "500K": dds_dir / "dds_results_500K.npy",
    }

    for name, path in dds_files.items():
        if path.exists():
            info = validate_dds_file(str(path))
            print(f"  ✓ {name}: shape={info.get('shape')}, dtype={info.get('dtype')}")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")

    # Step 2: Sample evaluation set
    print("\n[2/4] Preparing evaluation set...")
    eval_output = dds_dir / "dds_results_100K_eval.npy"
    eval_info = None

    if eval_output.exists() and not args.skip_eval_sampling:
        print(f"  ⚠ Eval file exists: {eval_output}")
        print("    Use --skip-eval-sampling to keep existing, or delete to regenerate")

    if not args.skip_eval_sampling:
        source_500k = dds_dir / "dds_results_500K.npy"
        if source_500k.exists():
            eval_info = prepare_dds_eval(
                str(source_500k),
                str(eval_output),
                n_samples=args.eval_samples,
                seed=args.seed,
            )
            print(f"  ✓ Created: {eval_output}")
            print(f"    Samples: {eval_info['n_samples']}, Seed: {eval_info['seed']}")
            print(f"    Indices SHA256: {eval_info['indices_sha256'][:16]}...")
        else:
            print(f"  ✗ Cannot create eval set: {source_500k} not found")
    else:
        print("  → Skipped (--skip-eval-sampling)")

    # Step 3: Validate OpenSpiel data (with format checking)
    print("\n[3/4] Validating OpenSpiel data...")
    openspiel_dir = raw_dir / "openspiel_bridge"
    openspiel_files = {
        "train": openspiel_dir / "train.txt",
        "test": openspiel_dir / "test.txt",
    }

    for name, path in openspiel_files.items():
        if path.exists():
            info = validate_openspiel_file(str(path))
            print(f"  ✓ {name}: {info.get('line_count'):,} lines")

            # Show format validation results
            fmt = info.get("format_validation", {})
            if fmt:
                print(f"    Format check ({fmt.get('n_sampled', 0)} samples):")
                print(f"      Features=480: {fmt.get('n_correct_features_480', 0)}/{fmt.get('n_parseable', 0)}")
                print(f"      Action∈[0,37]: {fmt.get('n_valid_action_0_37', 0)}/{fmt.get('n_parseable', 0)}")
                action_range = fmt.get('action_range', [0, 0])
                print(f"      Action range: [{action_range[0]}, {action_range[1]}]")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")

    # Step 4: Generate manifest (with lineage info)
    print("\n[4/4] Generating data manifest...")
    manifest = generate_data_manifest(str(project_root), eval_info=eval_info)

    manifest_path = raw_dir / "data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✓ Saved: {manifest_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Summary:")

    # DDS summary
    dds_ok = sum(1 for k, v in manifest["dds_data"].items() if v.get("exists", False))
    print(f"  DDS files: {dds_ok}/{len(manifest['dds_data'])} OK")

    # OpenSpiel summary
    os_ok = sum(1 for k, v in manifest["openspiel_data"].items() if v.get("exists", False))
    print(f"  OpenSpiel files: {os_ok}/{len(manifest['openspiel_data'])} OK")

    # Eval set
    if "dds_results_100K_eval.npy" in manifest.get("derived_data", {}):
        print(f"  Eval set: ✓ Ready")
    else:
        print(f"  Eval set: ✗ Not created")

    print("=" * 60)

    return 0 if dds_ok == len(dds_files) and os_ok == len(openspiel_files) else 1


if __name__ == "__main__":
    sys.exit(main())

"""
Ancient Writing Systems Analysis Pipeline
========================================
Full experimental pipeline treating Egyptian hieroglyphs and Sumerian cuneiform
as machine communication protocols / structured encoding systems.

Modules:
    1. entropy_analysis.py      - Shannon entropy analysis (H1, H2, redundancy)
    2. determinative_graph.py   - Classifier/determinative co-occurrence network
    3. geometric_phi_scanner.py - Golden ratio detection in sign geometry

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --entropy
    python run_pipeline.py --graph
    python run_pipeline.py --geometric
    python run_pipeline.py --corpus hieroglyphs --project etcsri
    python run_pipeline.py --image-scan
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODULES = {
    "entropy": "entropy_analysis.py",
    "graph": "determinative_graph.py",
    "geometric": "geometric_phi_scanner.py",
}


def run_module(module_name: str, corpus: str = "oracc", project: str = "etcsri",
               extra_args: list = None) -> dict:
    """Run a single analysis module and return results."""
    module_path = MODULES[module_name]

    # Build arguments per module
    if module_name == "geometric":
        # geometric module uses --signs not --corpus
        cmd = [sys.executable, module_path, "--signs", corpus]
    elif module_name == "entropy":
        cmd = [sys.executable, module_path, "--corpus", corpus, "--project", project]
    else:
        cmd = [sys.executable, module_path, "--corpus", corpus]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"RUNNING: {module_name.upper()}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=False
    )
    output = result.stdout + result.stderr

    # Try to load results
    results_file = None
    if module_name == "entropy":
        results_file = OUTPUT_DIR / "entropy" / "entropy_results.json"
    elif module_name == "graph":
        results_file = OUTPUT_DIR / "determinative_graph" / "determinative_graph_results.json"
    elif module_name == "geometric":
        results_file = OUTPUT_DIR / "geometric_phi" / "geometric_phi_results.json"

    results = {"output": output, "success": result.returncode == 0}

    if results_file and results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            results["data"] = json.load(f)

    return results


def run_full_pipeline(corpus: str = "oracc", project: str = "etcsri",
                      image_scan: bool = False) -> dict:
    """Run all three analysis modules in sequence."""
    print(f"\n{'='*60}")
    print("# ANCIENT WRITING SYSTEMS: FULL ANALYSIS PIPELINE")
    print(f"# Corpus: {corpus}")
    print(f"# Project: {project}")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "corpus": corpus,
        "modules": {},
    }

    # 1. Entropy analysis
    print("\n[STEP 1/3] Entropy Analysis")
    entropy_results = run_module("entropy", corpus, project)
    all_results["modules"]["entropy"] = entropy_results

    # 2. Determinative graph
    print("\n[STEP 2/3] Determinative Graph Analysis")
    graph_results = run_module("graph", corpus, project)
    all_results["modules"]["graph"] = graph_results

    # 3. Geometric/phi scanner
    print("\n[STEP 3/3] Geometric Phi Scanner")
    geometric_results = run_module("geometric", corpus, project)
    all_results["modules"]["geometric"] = geometric_results

    # Summary
    print("\n" + "#" * 60)
    print("# PIPELINE COMPLETE - SUMMARY")
    print("#" * 60)

    if entropy_results.get("success"):
        entropy_data = entropy_results.get("data", {})
        comparison = entropy_data.get("comparison", {})
        print(f"\nEntropy Classification: {comparison.get('classification', 'N/A')}")
        print(f"Closest Baseline: {comparison.get('closest_baseline', 'N/A')}")

    if graph_results.get("success"):
        graph_data = graph_results.get("data", {})
        interpretation = graph_data.get("interpretation", "N/A")
        print(f"\nDeterminative Graph: {interpretation}")

    if geometric_results.get("success"):
        geo_data = geometric_results.get("data", {})
        scan = geo_data.get("scan_result", {})
        print(f"\nPhi Scan: {scan.get('n_phi_matches', 0)} matches out of {scan.get('n_signs_scanned', 0)} signs")
        print(f"Overall Phi Score: {scan.get('overall_phi_score', 0):.3f}")

    # Save combined results
    combined_file = OUTPUT_DIR / "full_pipeline_results.json"
    with open(combined_file, 'w', encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Saved] Combined results to {combined_file}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ancient writing systems analysis pipeline")
    parser.add_argument("--module", choices=["entropy", "graph", "geometric"], default=None,
                        help="Run specific module only (default: all)")
    parser.add_argument("--corpus", choices=["oracc", "hieroglyphs"], default="oracc",
                        help="Corpus to analyze")
    parser.add_argument("--project", default="etcsri",
                        help="ORACC project name (e.g. etcsri, rinap, cams/gkab)")
    parser.add_argument("--all", action="store_true", help="Run all modules")
    parser.add_argument("--image-scan", action="store_true",
                        help="Run image-based phi scanning (requires images in data/hieroglyphs/)")

    args = parser.parse_args()

    if args.image_scan:
        from geometric_phi_scanner import run_image_phi_scan
        output_dir = PROJECT_ROOT / "outputs" / "geometric_phi"
        img_dir = PROJECT_ROOT / "data" / "hieroglyphs"
        run_image_phi_scan(img_dir=img_dir, output_dir=output_dir)
        return

    if args.module:
        run_module(args.module, args.corpus, args.project)
    else:
        run_full_pipeline(args.corpus, args.project)

"""Local artifact paths under repo outputs/. Cluster runs use $SCRATCH/prefix_caching (pipeline_config.sh)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
MUTATION_DIR = OUTPUTS_DIR / "mutation"
BENCHMARK_RESULTS_DIR = OUTPUTS_DIR / "benchmark_results"
PROMPT_ORGANIZATION_DIR = OUTPUTS_DIR / "prompt_organization"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
PROCESSED_DIR = OUTPUTS_DIR / "processed"

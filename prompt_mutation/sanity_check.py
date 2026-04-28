
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from data_loader import DataLoadConfig, load_examples
from prompt_generator import AlgorithmicMeaningChangingMutator
from overlap_analyzer import OverlapAnalyzer

examples = load_examples(DataLoadConfig(workload="scientific", max_samples=2))
mutator = AlgorithmicMeaningChangingMutator(seed=42)
record = mutator.mutate_scientific(examples[0], mutation_type="parameter_change")
metrics = OverlapAnalyzer().analyze(record.base_prompt, record.mutated_prompt)
print("OK", metrics.token_shared_prefix, record.metadata["changed_field"])

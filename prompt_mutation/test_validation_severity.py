
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .mutation_validation import ValidationConfig, validate_record
from .severity_calibration import SeverityConfig, SeverityCalibrator
from .prompt_generator import PromptRecord


def load_records(path: str) -> List[PromptRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(PromptRecord(**row))
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", required=True)
    p.add_argument("--validation-backend", default="sentence_transformer")
    p.add_argument("--validation-sentence-model-name", default="all-mpnet-base-v2")
    p.add_argument("--validation-nli-model-name", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    p.add_argument("--validation-bert-score-model-type", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--severity-backend", default="sentence_transformer")
    p.add_argument("--severity-sentence-model-name", default="all-mpnet-base-v2")
    p.add_argument("--severity-nli-model-name", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    p.add_argument("--severity-bert-score-model-type", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--output-path", default=None)
    args = p.parse_args()

    records = load_records(args.input_path)
    vcfg = ValidationConfig(
        semantic_backend=args.validation_backend,
        sentence_model_name=args.validation_sentence_model_name,
        nli_model_name=args.validation_nli_model_name,
        bert_score_model_type=args.validation_bert_score_model_type,
    )
    scfg = SeverityConfig(
        semantic_backend=args.severity_backend,
        sentence_model_name=args.severity_sentence_model_name,
        nli_model_name=args.severity_nli_model_name,
        bert_score_model_type=args.severity_bert_score_model_type,
    )
    calibrator = SeverityCalibrator(scfg)

    for rec in records:
        rec.metadata["validation"] = validate_record(rec, vcfg).to_dict()
        rec.metadata["severity"] = calibrator.measure(rec).to_dict()

    output_path = args.output_path or str(Path(args.input_path).with_name(Path(args.input_path).stem + ".validated.jsonl"))
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved validated dataset to: {output_path}")


if __name__ == "__main__":
    main()

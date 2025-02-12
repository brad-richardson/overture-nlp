import argparse
import csv
import json
import pathlib
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import pandas as pd

from llama_cpp import Llama

# 3.3 70B - needs large GPU, likely not runnable locally
# MODEL_REPO = "bartowski/Llama-3.3-70B-Instruct-GGUF"
# MODEL_FILENAME = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"

# 3.1 8B
MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# 3.2 3B
# MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
# MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"


def create_lv_system_prompt() -> str:
    # TODO - this needs work
    return f"""
    Assess whether the provided name for a crowd-sourced map feature from OpenStreetMap is correctly labeled as specifically for English map users.

    Please provide a label of either `no_issue` or `invalid`. Then, give it a risk score between 0.0 and 1.0, where 0.0 means this is definitely intended for English readers on the map and 1.0 means not an English name and could not possibly be interpreted as such. Also include a brief paragraph describing reasoning.

    An example of `no_issue` would be "Germany", whereas "Deutschland" would be `invalid`. Default to `no_issue` when unsure.
    """


def evaluate_language_validation(llm):
    # TODO - this needs work
    system_prompt = create_lv_system_prompt()
    allowed_labels = ["no_issue", "invalid"]
    json_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": allowed_labels},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
    }
    user_prompt = "TODO"

    try:
        generated = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
            temperature=0.2,
            top_p=0.9,
            max_tokens=1024,
        )

        response = json.loads(generated.get("choices")[0].get("message").get("content"))

        return response
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def create_vandalism_system_prompt() -> str:
    return """
    You are a senior map QA analyst, tasked with reviewing name changes to prevent offensive content from being shown to end users.

    These feature names are crowd-sourced from OpenStreetMap and could contain vandalism, profanity, vulgarities or suspicious changes (typos, test data, names from fictional works).
    Note that these names may be in any language, not just English. During assessment, please consider diverse political, lingual, socioconomic and geographic viewpoints in the evaluation.

    Please provide the following:
    - a label of "vandalism", "profanity", "suspicious" or "no_issue"
    - a risk score between 0.0 and 1.0, where 0.0 indicates the name has no chance of being problematic and could appear on e.g. a car nav system and 1.0 means definitely an issue and likely to be offensive or unwelcome
    - a brief (2 sentences) reasoning for the rating

    Additional prompt context may be provided, such as tag key (indicating language or special name type), previous name and the feature's OSM tags as JSON. These fields should be used as context only, and not directly considered for the output label and score.
    """


def evaluate_vandalism(
    llm,
    new_name: Optional[str],
    old_name: Optional[str],
    name_key: Optional[str],
    filtered_tags_json: Optional[str],
) -> Optional[dict]:
    system_prompt = create_vandalism_system_prompt()

    # Build user prompt
    if not new_name:
        user_prompt = "Name was removed\n"
    else:
        user_prompt = f"New name: '{new_name}'\n"
    if old_name:
        user_prompt += f"Old name: '{old_name}'\n"
    if name_key:
        user_prompt += f"OSM name key: '{name_key}'\n"
    if filtered_tags_json:
        user_prompt += f"OSM tags: {filtered_tags_json}\n"

    allowed_labels = ["vandalism", "profanity", "suspicious", "no_issue"]
    json_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": allowed_labels},
            "risk_score": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["label", "risk_score", "reason"],
    }

    generated = None
    try:
        generated = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
            # Low temperature for predictability, high top_p for diversity
            temperature=0.2,
            top_p=0.9,
        )

        raw_response = generated.get("choices")[0].get("message").get("content")
        response = json.loads(raw_response)

        response["prompt"] = user_prompt
        response["new_name"] = new_name
        response["old_name"] = old_name
        return response
    except Exception as e:
        print(f"Error processing row: {e}")
        if generated:
            print(generated)
        return None

@dataclass
class EvaluationResult:
    false_negatives: int = 0
    false_positives: int = 0
    true_negative: int = 0
    true_positive: int = 0
    evaluations: List[Dict] = field(default_factory=lambda: list())

    def increment_false_negative(self, amount: int = 1):
        self.false_negatives += amount

    def increment_false_positive(self, amount: int = 1):
        self.false_positives += amount

    def increment_true_negative(self, amount: int = 1):
        self.true_negative += amount

    def increment_true_positive(self, amount: int = 1):
        self.true_positive += amount
    
    def add_evaluation(self, result: dict):
        self.evaluations.append(result)

    @property
    def total_samples(self) -> int:
        return (self.false_negatives + self.false_positives + 
                self.true_negative + self.true_positive)

    @property
    def accuracy(self) -> float:
        """
        Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
        Returns:
            float: Accuracy score between 0 and 1
        """
        total = self.total_samples
        if total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / total

    @property
    def precision(self) -> float:
        """
        Calculate precision: TP / (TP + FP)
        Returns:
            float: Precision score between 0 and 1
        """
        denominator = self.true_positive + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator

    @property
    def recall(self) -> float:
        """
        Calculate recall: TP / (TP + FN)
        Returns:
            float: Recall score between 0 and 1
        """
        denominator = self.true_positive + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator

    @property
    def f1_score(self) -> float:
        """
        Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        Returns:
            float: F1 score between 0 and 1
        """
        precision = self.precision
        recall = self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def __str__(self) -> str:
        """Pretty string representation of metrics"""
        return (
            f"Evaluation Metrics:\n"
            f"  Total Samples: {self.total_samples}\n"
            f"  Accuracy:  {self.accuracy:.3f}\n"
            f"  Precision: {self.precision:.3f}\n"
            f"  Recall:    {self.recall:.3f}\n"
            f"  F1 Score:  {self.f1_score:.3f}\n"
            f"  Counts:\n"
            f"    True Positives:  {self.true_positive}\n"
            f"    True Negatives:  {self.true_negative}\n"
            f"    False Positives: {self.false_positives}\n"
            f"    False Negatives: {self.false_negatives}"
        )

def evaluate_prompts(eval_type: str, csv_path: str) -> EvaluationResult:
    print(f"Loading model {MODEL_FILENAME}")
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        n_ctx=2048,
        verbose=False,
    )
    print(f"Loaded model {MODEL_FILENAME}")

    # Read CSV file
    # pd.read_csv()
    with open(csv_path, "rb") as f:
        num_lines = sum(1 for _ in f)

    with open(csv_path, "r") as f:
        # df = pd.read_csv(f)
        reader = csv.DictReader(f, delimiter=",")

        eval_result = EvaluationResult()

        index = 0
        for row in reader:
            index += 1

            # Print progress every 10%
            if index % round(num_lines / 10) == 0 and index > 0:
                print(f"{datetime.now()}: At index {index}")

            # Run evaluation
            if eval_type == "language-validation":
                evaluation = evaluate_language_validation(llm)
            elif eval_type == "vandalism":
                evaluation = evaluate_vandalism(
                    llm,
                    new_name=row.get("new_name"),
                    old_name=row.get("old_name"),
                    name_key=row.get("tag"),
                    filtered_tags_json=row.get("filtered_tags"),
                )
            else:
                raise ValueError(f"Unknown eval type: {eval_type}")

            if evaluation:
                if "label" in row:
                    evaluation["expected_label"] = row.get("label")

                    predicted_issue = evaluation["label"] != "no_issue"
                    expected_issue = evaluation["expected_label"] not in ["no_issue", "not_an_issue"]
                    if predicted_issue and expected_issue:
                        eval_result.increment_true_positive()
                    elif not predicted_issue and not expected_issue:
                        eval_result.increment_true_negative()
                    elif predicted_issue and not expected_issue:
                        eval_result.increment_false_positive()
                    else:
                        eval_result.increment_false_negative()
                
                eval_result.add_evaluation(evaluation)

    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Overture NLP")

    parser.add_argument("--type", choices=["language-validation", "vandalism"])
    parser.add_argument("--out", default="./output")
    parser.add_argument("--input-csv")

    args = parser.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    input_csv = args.input_csv

    t0 = time.time()
    eval_result = evaluate_prompts(eval_type=args.type, csv_path=input_csv)
    t1 = time.time()

    print(f"Total time (sec): {(t1-t0):.2f}")
    model_results_path = f"{args.out}/{args.type}.json"
    with open(model_results_path, "w") as f:
        json.dump(eval_result.evaluations, f, indent=4)
    if eval_result.total_samples > 0:
        print(eval_result)
    

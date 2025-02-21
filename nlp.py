import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pathlib
import random
import time
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field

import pandas as pd


class EvalBackend:
    def set_model_name(self, model_name: str):
        self.model_name = model_name

    def set_openai_clients(self, clients):
        self.local_llm = None
        self.openai_clients = clients

    def set_llama_model(self, model):
        self.local_llm = model
        self.openai_clients = None

    def create_chat_completion(
        self, system_prompt: str, user_prompt: str, json_schema: dict
    ) -> dict:
        if self.local_llm:
            generated = self.local_llm.create_chat_completion(
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
            )
            raw_response = generated.get("choices")[0].get("message").get("content")
            return json.loads(raw_response)
        elif self.openai_clients:
            # Very naive round robin using random indices
            client_index = random.randrange(0, len(self.openai_clients))
            generated = self.openai_clients[client_index].chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_object",
                    "schema": json_schema,
                },
                temperature=0.2,
            )
            raw_response = generated.choices[0].message.content
            return json.loads(raw_response)
        else:
            raise RuntimeError("No eval backend setup")


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
        return (
            self.false_negatives
            + self.false_positives
            + self.true_negative
            + self.true_positive
        )

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
            json_schema=json_schema,
        )

        response = json.loads(generated.get("choices")[0].get("message").get("content"))

        return response
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def create_vandalism_system_prompt(generate_reasoning: bool = False) -> str:
    return f"""
    You are a senior map QA assistant, tasked with reviewing name changes to prevent offensive content from being shown to end users.

    These feature name changes are crowd-sourced from OpenStreetMap and could contain vandalism, profanity, vulgarities or suspicious changes (typos, test or private data, names from fictional works).
    Note that these names may be in any language, not just English. During assessment, please consider diverse political, lingual, socioconomic and geographic viewpoints in the evaluation.

    Please provide the following:
    - label of "vandalism", "profanity", "suspicious" or "no_issue"
    - risk score between 0.0 and 1.0, where 0.0 indicates the name has no chance of being problematic and could appear on e.g. a car nav system and 1.0 means definitely an issue and generally offensive or unwelcome
    {"- a brief sentence reasoning for the rating" if generate_reasoning else ""}

    Additional context may be provided in the request (such as OSM tags) and that information should be used as context only, not evaluated directly for the output label and score.
    """


def evaluate_vandalism(
    eval_backend: EvalBackend,
    system_prompt: str,
    new_name: Optional[str],
    old_name: Optional[str],
    name_key: Optional[str],
    filtered_tags_json: Optional[str],
    additional_columns: Optional[Dict],
    generate_reasoning: bool = False,
) -> Optional[dict]:
    # Build user prompt
    if not new_name:
        user_prompt = "Name was removed"
    else:
        user_prompt = f"New name: '{new_name}'"
    if old_name:
        user_prompt += f"\nOld name: '{old_name}'"
    if name_key:
        user_prompt += f"\nOSM tag key: '{name_key}'"
    if filtered_tags_json:
        user_prompt += f"\nOSM tags: {filtered_tags_json}"

    allowed_labels = ["vandalism", "profanity", "suspicious", "no_issue"]
    json_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": allowed_labels},
            "risk_score": {"type": "number"},
        },
        "required": ["label", "risk_score"],
    }
    if generate_reasoning:
        # Adds significant overhead to generation, default to not include this
        json_schema["properties"]["reason"] = {"type": "string"}
        json_schema["required"].append("reason")

    response = {}
    try:
        response = eval_backend.create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
        )
    except Exception as e:
        response["error"] = f"Error processing row: {e}"

    if generate_reasoning:
        response["prompt"] = user_prompt
    response["new_name"] = new_name
    response["old_name"] = old_name
    if additional_columns:
        for col, val in additional_columns.items():
            if val:
                response[col] = val
    return response


def evaluate_prompts(
    input_csv: str,
    input_parquet: str,
    eval_type: str,
    backend: str,
    model_repo: str,
    model_name: str,
    server_urls: str,
    output_path: str,
    threads: int,
    generate_reasoning: bool,
) -> EvaluationResult:

    eval_backend = EvalBackend()
    if backend == "llama-cpp":
        import llama_cpp

        print(f"Loading model {model_name}")
        llm = llama_cpp.Llama.from_pretrained(
            repo_id=model_repo,
            filename=model_name,
            n_ctx=4096,
            verbose=False,
        )
        eval_backend.set_llama_model(llm)
        print(f"Loaded model {model_name}")

        # Not thread-safe
        threads = 1
    elif backend == "llama-cpp-server":
        import openai

        server_urls = server_urls.split(",")
        clients = []
        for server_url in server_urls:
            openai_client = openai.OpenAI(
                base_url=server_url, api_key="sk-no-key-required"
            )
            clients.append(openai_client)
        eval_backend.set_openai_clients(clients)
        eval_backend.set_model_name(model_name)
    else:
        raise ValueError(f"Invalid backend option: {backend}")

    if input_csv:
        df = pd.read_csv(input_csv)
    elif input_parquet:
        df = pd.read_parquet(input_parquet)
    else:
        raise ValueError("Need input CSV or Parquet")

    row_count = len(df)

    eval_result = EvaluationResult()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        system_prompt = create_vandalism_system_prompt(
            generate_reasoning=generate_reasoning
        )

        # Submit all tasks
        for _, row in df.iterrows():
            if eval_type == "language-validation":
                # TODO - update this, handle concurrency
                # evaluation = evaluate_language_validation(eval_backend)
                pass
            elif eval_type == "vandalism":
                future = executor.submit(
                    evaluate_vandalism,
                    eval_backend=eval_backend,
                    system_prompt=system_prompt,
                    new_name=row.get("new_name"),
                    old_name=row.get("old_name"),
                    name_key=row.get("tag"),
                    filtered_tags_json=row.get("filtered_tags"),
                    generate_reasoning=generate_reasoning,
                    additional_columns={
                        "expected_label": row.get("label", ""),
                        "osm_id": row.get("osm_id", ""),
                        "osm_type": row.get("osm_type", ""),
                        "version": row.get("version", ""),
                    },
                )
                futures.append(future)
            else:
                raise ValueError(f"Unknown eval type: {eval_type}")

        # Collect results as they complete
        model_results_path = f"{output_path}/{eval_type}-ongoing-{datetime.now()}.json"
        with open(model_results_path, "w") as f:
            for future in futures:
                try:
                    evaluation = future.result()

                    if not evaluation:
                        continue

                    if "expected_label" in evaluation:
                        predicted_issue = evaluation.get("label", "") != "no_issue"
                        expected_issue = evaluation["expected_label"] not in [
                            "no_issue",
                            "not_an_issue",
                        ]
                        if predicted_issue and expected_issue:
                            eval_result.increment_true_positive()
                        elif not predicted_issue and not expected_issue:
                            eval_result.increment_true_negative()
                        elif predicted_issue and not expected_issue:
                            eval_result.increment_false_positive()
                        else:
                            eval_result.increment_false_negative()

                    eval_result.add_evaluation(evaluation)

                    f.writelines([json.dumps(evaluation) + "\n"])
                except Exception as e:
                    print(f"TODO - failure grabbing result, {e}")

                # Print progress every 1%
                if len(eval_result.evaluations) % max(1, round(row_count / 100)) == 0:
                    print(
                        f"{datetime.now()}: Evaluated {len(eval_result.evaluations)}/{row_count}"
                    )

    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Overture NLP")

    parser.add_argument("--input-csv")
    parser.add_argument("--input-parquet")
    parser.add_argument("--type", choices=["language-validation", "vandalism"])
    parser.add_argument("--output", default="./output")
    # llama-cpp uses python library directly and is the easiest method for macOS
    # llama-cpp-server can point at a locally running service (recommended to use docker)
    parser.add_argument(
        "--backend",
        choices=["llama-cpp", "llama-cpp-server"],
        default="llama-cpp",
        help="whether to manage model directly (easier to get started, serial) or use server (parallel)",
    )
    # Multiple instances needed for llama.cpp server, as cpu can be bottleneck
    # See https://github.com/ollama/ollama/issues/7648#issuecomment-2473561990 for more details
    parser.add_argument(
        "--server-urls",
        default="http://localhost:8080/v1",
        help="comma-separated list of server URLs to rotate between",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--model-repo", default="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    )
    parser.add_argument(
        "--model-name", default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    # Verify able to write to path and pyarrow installed
    results_path = f"{args.output}/{args.type}-output.parquet"
    pd.DataFrame([{"placeholder": "data"}]).to_parquet(results_path)

    t0 = time.time()
    eval_result = evaluate_prompts(
        input_csv=args.input_csv,
        input_parquet=args.input_parquet,
        eval_type=args.type,
        backend=args.backend,
        server_urls=args.server_urls,
        threads=args.threads,
        model_repo=args.model_repo,
        model_name=args.model_name,
        output_path=args.output,
        generate_reasoning=args.debug,
    )
    t1 = time.time()
    print(f"Total time (sec): {(t1-t0):.2f}")
    if eval_result.total_samples > 0:
        print(eval_result)

    output_df = pd.DataFrame(eval_result.evaluations)
    output_df.to_parquet(results_path)

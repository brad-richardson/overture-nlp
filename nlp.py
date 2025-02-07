import argparse
import csv
import json
import pathlib
import time
from datetime import datetime

from llama_cpp import Llama

# MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
# MODEL_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"

MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q6_K_L.gguf"


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
    - a label of "profanity", "vandalism", "suspicious" or "no_issue"
    - a risk score between 0.0 and 1.0, where 0.0 indicates the name has no chance of being problematic and could appear on e.g. a car nav system and 1.0 means definitely an issue and likely to be offensive or unwelcome
    - a brief (<=2 sentences) reasoning for the rating

    Additional prompt context may be provided, such as language code, previous name and the feature's OSM tags as JSON. These fields should be used as context only, and not directly considered for the output label and score.
    """


def evaluate_vandalism(
    llm, new_name, old_name, target_language_code, filtered_tags_json
):
    system_prompt = create_vandalism_system_prompt()

    # Build user prompt
    if not new_name:
        user_prompt = "Name was removed\n"
    else:
        user_prompt = f"New name: '{new_name}'\n"
    if target_language_code:
        user_prompt += f"Target language code: '{target_language_code}'\n"
    if old_name:
        user_prompt += f"Old name: '{old_name}'\n"
    if filtered_tags_json:
        user_prompt += f"Tags: {filtered_tags_json}\n"

    allowed_labels = ["profanity", "vandalism", "suspicious", "no_issue"]
    json_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": allowed_labels},
            "risk_score": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["label", "risk_score", "reason"],
    }

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
            max_tokens=2048,
        )

        response = json.loads(generated.get("choices")[0].get("message").get("content"))

        response["new_name"] = new_name
        response["old_name"] = old_name
        return response
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def evaluate_prompts(csv_path, eval_type):
    print(f"Loading model {MODEL_FILENAME}")
    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        verbose=False,
    )
    print(f"Loaded model {MODEL_FILENAME}")

    # Read CSV file
    with open(csv_path, "rb") as f:
        num_lines = sum(1 for _ in f)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=",")

        correct = 0
        results = []

        index = 0
        for row in reader:
            index += 1

            # Print progress every 10%
            if index % round(num_lines / 10) == 0 and index > 0:
                print(f"{datetime.now()}: At index {index}")

            # Run evaluation
            if eval_type == "language-validation":
                result = evaluate_language_validation(llm)
            elif eval_type == "vandalism":
                # TODO - remaining params
                result = evaluate_vandalism(llm, row.get("name_tag"), "", "", "")
                if result:
                    result["expected"] = row.get("label")
            else:
                raise ValueError(f"Unknown eval type: {eval_type}")

            if result:
                results.append(result)
                if result["label"] != "no_issue" and result["expected"] != "regular":
                    correct += 1
                elif result["label"] == "no_issue" and result["expected"] == "regular":
                    correct += 1

    return results, correct / index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Overture NLP")

    parser.add_argument("filename")
    parser.add_argument("--type", choices=["language-validation", "vandalism"])
    parser.add_argument("--log-dir", default="./logs")

    args = parser.parse_args()

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results, accuracy = evaluate_prompts(args.filename, args.type)
    t1 = time.time()

    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Total time (min): {(t1-t0)/60.0:.2}")
    model_log_path = f"{args.log_dir}/{args.type}.json"
    with open(model_log_path, "w") as f:
        json.dump(results, f, indent=4)

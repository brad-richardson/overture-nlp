from ollama import Client
import json
import time
import argparse
import pathlib
import csv

def create_lv_prompt(name_to_test: str) -> str:
    return f"""
    Assess whether the crowd-sourced map feature from OpenStreetMap with the name "{name_to_test}" is correctly labeled as specifically for English map users.

    Please provide a label of either `no_issue` or `invalid`. Then, give it a risk score between 0.0 and 1.0, where 0.0 means this is definitely intended for English readers on the map and 1.0 means not an English name and could not possibly be interpreted as such. Also include a brief paragraph describing reasoning.

    An example of `no_issue` would be "Germany", whereas "Deutschland" would be `invalid`. Default to `no_issue` when unsure.
    """

def create_vandalism_prompt(name_to_test: str) -> str:
    return f"""
    Assess whether the crowd-sourced map feature from OpenStreetMap with the name "{name_to_test}" is vandalism or profane or would not be used to describe real world features like roads, buildings, places or POIs. Note that these can be in any world language, not just English.

    Please provide a label of either "vandalism" or "regular", rate it between 0.0 and 1.0, where 0.0 means a name that has no chance of being vandalism and could appear on e.g. a car nav system and 1.0 means this is very likely an invalid or vandalized name in this context. Also a brief paragraph describing reasoning.
    """

def evaluate_prompts(model_name, csv_path, eval_type):
    # Initialize local Ollama client
    client = Client()

    # Read CSV file
    reader = csv.DictReader(open(csv_path), delimiter=',')

    # Initialize results tracking
    correct = 0
    results = []

    idx = 0
    for row in reader:
        idx += 1

        if idx % 50 == 0 and idx > 0:
            print(f"At {idx}, accuracy: {correct/(idx):.2%}")

        if eval_type == "language-validation":
            prompt = create_lv_prompt(row.get("name_tag"))
            allowed_labels = ["no_issue", "invalid"]
        elif eval_type == "vandalism":
            prompt = create_vandalism_prompt(row.get("name_tag"))
            allowed_labels = ["regular", "vandalism"]

        try:
            # Generate response using Ollama
            generated = client.generate(
                model=model_name,
                prompt=prompt,
                stream=False,
                format={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": allowed_labels},
                        "risk_score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "risk_score", "reason"],
                },
            )

            response = json.loads(generated.response)

            if response.get("label", "") == row["label"]:
                correct += 1
            elif response.get("label", "") not in allowed_labels:
                raise RuntimeError("Invalid label provided")
            else:
                # print(f"Mismatch for {row['name_tag']} | expected {row['label']} | score {response.get("risk_score")}")
                pass
            
            response["name_tag"] = row.get("name_tag")
            response["expected"] = row.get("label")
            results.append(response)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            exit(1)

    return results, correct / (idx + 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Overture NLP')
    
    parser.add_argument("filename")
    parser.add_argument("--type", choices=['language-validation', 'vandalism'])
    parser.add_argument("--model")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--log-dir", default="./logs")

    args = parser.parse_args()

    if not args.all_models and not args.model:
        print("Need --model or --all-models set")
        exit(1)
    
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    models = ["llama3.2:1b", "llama3.2:3b", "phi4:14b", "deepseek-r1:14b"] if args.all_models else [args.model]

    print(f"Evaluating {args.type} for models {models}")
    for model in models:
        print(f"\nModel {model}")

        t0 = time.time()
        results, accuracy = evaluate_prompts(model, args.filename, args.type)
        t1 = time.time()

        print(f"Overall Accuracy: {accuracy:.2%}")
        print(f"Total time (min): {(t1-t0)/60.0:.2}")
        model_log_path = f"{args.log_dir}/vandalism_{model.replace(":", "")}.json"
        with open(model_log_path, 'w') as f:
            json.dump(results, f, indent=4)

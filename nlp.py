import pandas as pd
from ollama import Client
import json

VANDALISM_THRESHOLD = 0.5
MODEL_NAME = "llama3.2:3b"
# llama3.2:3b - Overall Accuracy: 60.00%
# phi4 - Overall Accuracy: 70.00%
# llama3.3:70b
INPUT_FILE = "vandalism-sample.csv"
OUTPUT_FILE = "out.json"


def create_prompt(name_to_test: str) -> str:
    return f"""
    Assess whether the crowd-sourced map feature from OpenStreetMap with the name "{name_to_test}" is vandalism or profane or would not be used to describe real world features like roads, buildings, places or POIs. Note that these can be in any world language, not just English.

    Please rate it between 0.0 and 1.0, where 0.0 means a name that has no chance of being vandalism and could appear on e.g. a car nav system and 1.0 means this is very likely an invalid or vandalized name in this context. Please briefly explain reasoning in a paragraph as well.
    """


def evaluate_prompts(csv_path):
    # Initialize local Ollama client
    client = Client()

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Initialize results tracking
    correct = 0
    total = len(df)
    mismatches = []

    for idx, row in df.iterrows():

        if idx % 5 == 0 and idx > 0:
            print(f"At {idx}/{total}, accuracy: {correct/(idx):.2%}")

        prompt = create_prompt(row["name_tag"])

        try:
            # Generate response using Ollama
            generated = client.generate(
                model=MODEL_NAME,
                prompt=prompt,
                stream=False,
                format={
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                },
            )

            response = json.loads(generated.response)
            if response.get("score") >= VANDALISM_THRESHOLD and row["label"] == "vandalism":
                correct += 1
            elif response.get("score") < VANDALISM_THRESHOLD and row["label"] != "vandalism":
                correct += 1
            else:
                print(f"Mismatch for {row['name_tag']} | expected {row['label']} | score {response.get("score")}")
                mismatches.append(response)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            exit(1)

    return mismatches, correct / total


if __name__ == "__main__":
    mismatches, accuracy = evaluate_prompts(INPUT_FILE)

    # Print results
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(mismatches, f, indent=4)

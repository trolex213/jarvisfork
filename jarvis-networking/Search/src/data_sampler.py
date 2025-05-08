import json
from pathlib import Path

def sample_jsonl(file_path: Path, num_samples: int = 5):
    samples = []
    with file_path.open() as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error in line {i+1}: {e}")
    return samples

if __name__ == "__main__":
    samples = sample_jsonl(Path("normalized_data.json"))
    print(json.dumps(samples, indent=2))

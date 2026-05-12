import json

with open("cleaned_output.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

converted = []

for row in raw:
    converted.append({
        "question": row["q"],
        "reference": row["a"]
    })

with open("data/ablation_dataset.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print("Done")
import json

file_path = "D:/Projects/donut/data/train.json"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

valid_lines = []
errors_found = False

for i, line in enumerate(lines):
    try:
        sample = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"❌ Line {i+1}: Invalid JSON - {e}")
        errors_found = True
        continue

    if "image" not in sample:
        print(f"❌ Line {i+1}: Missing 'image' key.")
        errors_found = True
        continue

    if "ground_truth" not in sample:
        print(f"❌ Line {i+1}: Missing 'ground_truth' key.")
        errors_found = True
        continue

    try:
        gt_obj = json.loads(sample["ground_truth"])
        if "gt_parse" not in gt_obj:
            print(f"❌ Line {i+1}: 'ground_truth' exists but missing 'gt_parse'.")
            errors_found = True
            continue
    except json.JSONDecodeError as e:
        print(f"❌ Line {i+1}: 'ground_truth' is not a valid JSON string - {e}")
        errors_found = True
        continue

    valid_lines.append(sample)

print(f"\n✅ {len(valid_lines)} valid entries found out of {len(lines)}.")

# Optional: Overwrite with only valid entries
if not errors_found:
    print("🎉 All entries are valid! You can now run training.")
else:
    fix = input("Would you like to save only valid entries to a new file? (y/n): ").lower()
    if fix == "y":
        with open("D:/Projects/donut/data/train_cleaned.json", "w", encoding="utf-8") as f:
            for item in valid_lines:
                f.write(json.dumps(item) + "\n")
        print("✅ Cleaned data saved to train_cleaned.json")

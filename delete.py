

import json

TARGET = "000000000761"
FILE_PATH = "D:/COCO/annotations/person_keypoints_train2017.json"   # ← change this to your JSON file path


results = []

def search_json(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            search_json(value, f"{path}.{key}" if path else key)

    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            search_json(item, f"{path}[{index}]")

    else:
        # Convert non-string values to string for substring search
        try:
            text = str(obj)
            if TARGET in text:
                results.append(path)
        except:
            pass

# Load JSON
with open(FILE_PATH, "r") as f:
    data = json.load(f)

# Search
search_json(data)

# Print results
if results:
    print("Found substring matches at:")
    for r in results:
        print(" -", r)
else:
    print("No matches found.")
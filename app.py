import json
import pandas as pd
from pathlib import Path

# Change this to your actual file path
file_path = "data.jsonl"  # <-- UPDATE THIS

# Load JSON
with open(file_path, 'r') as f:
    data = json.load(f)

# Print basic info
print(f"Total records: {len(data)}")
print(f"\nData type of first element: {type(data[0])}")

# If it's a list of dicts (most common)
if isinstance(data, list) and len(data) > 0:
    first_item = data[0]
    print(f"\nKeys in each record: {list(first_item.keys())}")
    print(f"\nFirst record preview:")
    for key, value in first_item.items():
        value_preview = str(value)[:100] if value else "None"
        print(f"  {key}: {value_preview}")
    
    # Check for nested structures
    for key, value in first_item.items():
        if isinstance(value, dict):
            print(f"\n  Nested in '{key}': {list(value.keys())}")
        elif isinstance(value, list) and len(value) > 0:
            print(f"\n  '{key}' is a list with {len(value)} items")
            if isinstance(value[0], dict):
                print(f"    First item keys: {list(value[0].keys())}")

# If it's a dict of lists or different structure
elif isinstance(data, dict):
    print(f"\nTop-level keys: {list(data.keys())}")
    for key in list(data.keys())[:3]:
        print(f"  {key}: {type(data[key])}, length: {len(data[key]) if hasattr(data[key], '__len__') else 'N/A'}")

# Check for timestamps - critical for time series
print("\n" + "="*50)
print("LOOKING FOR TIME FIELDS:")
time_indicators = ['time', 'date', 'timestamp', 'created', 'published', 'posted']
for i, item in enumerate(data[:10]):  # Check first 10 records
    for key in item.keys():
        if any(indicator in key.lower() for indicator in time_indicators):
            print(f"Found potential time field: '{key}' = {item[key]}")
            break

# Check for text content
print("\nLOOKING FOR TEXT FIELDS:")
text_indicators = ['text', 'content', 'body', 'message', 'title', 'description', 'post']
for i, item in enumerate(data[:10]):
    for key in item.keys():
        if any(indicator in key.lower() for indicator in text_indicators):
            preview = str(item[key])[:80]
            print(f"Found text field: '{key}' = {preview}...")
            break

# Check for author/user fields
print("\nLOOKING FOR AUTHOR FIELDS:")
author_indicators = ['author', 'user', 'username', 'account', 'creator', 'from']
for i, item in enumerate(data[:10]):
    for key in item.keys():
        if any(indicator in key.lower() for indicator in author_indicators):
            print(f"Found author field: '{key}' = {item[key]}")
            break
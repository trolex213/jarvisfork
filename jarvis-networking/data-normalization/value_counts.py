import json
from collections import Counter
import pandas as pd

# Read the JSON file and count normalized company names
company_counts = Counter()
total_lines = 0
valid_lines = 0
error_lines = 0

print("Processing file...")
with open('summarized_profiles_with_normalized.json', 'r') as f:
    for line in f:
        total_lines += 1
        try:
            # Try to find the first complete JSON object in the line
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Find the first complete JSON object
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(line):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                json_str = line[:end_pos]
                profile = json.loads(json_str)
                if 'normalized_company_name' in profile and profile['normalized_company_name']:
                    company_counts[profile['normalized_company_name']] += 1
                    valid_lines += 1
        except json.JSONDecodeError as e:
            error_lines += 1
            continue
        except Exception as e:
            error_lines += 1
            continue

# Convert to DataFrame for better display
df = pd.DataFrame.from_dict(company_counts, orient='index', columns=['count'])
df = df.sort_values('count', ascending=False)

# Save to CSV
df.to_csv('normalized_company_counts.csv')

print(f"\nProcessing complete!")
print(f"Total lines processed: {total_lines}")
print(f"Valid profiles with company names: {valid_lines}")
print(f"Error lines: {error_lines}")
print(f"\nTop 20 most common normalized company names:")
print(df.head(20))
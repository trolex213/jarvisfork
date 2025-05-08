import json

# Process the file line by line and write formatted JSON
with open('summarized_profiles_with_normalized.json', 'r') as input_file, \
     open('summarized_profiles_with_normalized_formatted.json', 'w') as output_file:
    
    for line in input_file:
        try:
            # Parse and format each JSON object
            profile = json.loads(line.strip())
            formatted_json = json.dumps(profile, indent=2)
            # Write the formatted JSON and add a newline to separate entries
            output_file.write(formatted_json + '\n')
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {e}")
            continue

print("Formatting complete! New file created: summarized_profiles_with_normalized_formatted.json")
print("Verifying line count...") 
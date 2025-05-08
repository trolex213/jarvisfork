import json

# List of all variations of JP Morgan names
JPMORGAN_VARIATIONS = {
    "J.P. Morgan",
    "JPMorgan Chase & Co.",
    "JPMorganChase",
    "JPMorgan Chase",
    "JP Morgan",
    "JP Morgan Chase",
    "JPMorgan",
    "JP Morgan Asset Management",
    "jp morgan",
    "JP. Morgan",
    "JP Morgan Private Bank",
    "JPM Funds",
    "JPMC",
    "JPMorgan Chase & C"
}

# Function to normalize company names
def normalize_company_name(name):
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and extra spaces
    name = ' '.join(name.split())
    # Check if it's one of the JP Morgan variations
    if any(variation.lower() in name for variation in JPMORGAN_VARIATIONS):
        return "jp morgan"
    return name

# Read the input file
with open('summarized_profiles_with_normalized.json', 'r') as f:
    data = json.load(f)

# Process each profile
for profile in data:
    # Normalize current company name
    if 'current_company_name' in profile and profile['current_company_name'] is not None:
        profile['current_company_name'] = normalize_company_name(profile['current_company_name'])
    
    # Normalize normalized company name
    if 'normalized_company_name' in profile and profile['normalized_company_name'] is not None:
        profile['normalized_company_name'] = normalize_company_name(profile['normalized_company_name'])

# Write the updated data back to a new file
with open('summarized_profiles_with_normalized_jpmorgan.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Processing complete! Check the new file: summarized_profiles_with_normalized_jpmorgan.json")

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

# List of all variations of Goldman Sachs names
GOLDMAN_SACHS_VARIATIONS = {
    "Goldman Sachs",
    "Goldman Sachs & Co.",
    "Goldman Sachs & Co",
    "Goldman Sachs - Investment Banking - Cross Markets Group - Middle Market Leveraged Finance",
    "Goldman Sachs Ayco Personal Financial Management",
    "Goldman sachs"
}

# List of all variations of Bank of America names
BANK_OF_AMERICA_VARIATIONS = {
    "Bank of America",
    "Bank of America Merrill Lynch",
    "Merrill Lynch",
    "Merrill Lynch Wealth Management",
    "Bank of America - Merrill Lynch",
    "Merrill Lynch - Bank of America",
    "Merrill Lynch Global Wealth Management",
    "BofA Merrill Lynch",
    "Bank of America / Merrill Lynch",
    "Bank of America Merrill Lynch (MLPF&S)",
    "Bank of America Securities",
    "Bank of America-Merrill Lynch"
}

# List of all variations of Morgan Stanley names
MORGAN_STANLEY_VARIATIONS = {
    "Morgan Stanley",
    "Morgan Stanley Wealth Management",
    "Morgan Stanley Investment Management",
    "E*TRADE from Morgan Stanley",
    "Morgan Stanley Private Wealth Management",
    "Mitsubishi UFJ Morgan Stanley Securities",
    "Morgan Stanley & Co",
    "Morgan Stanley U.S. Banks",
    "Morgan Stanley Smith Barney",
    "Morgan Stanley Real Estate Investing",
    "Morgan Stanley Private Equity",
    "Morgan Stanley Capital Partners",
    "Morgan Stanley Capital International",
    "Graystone Consulting at Morgan Stanley"
}

# List of all variations of UBS names
UBS_VARIATIONS = {
    "UBS",
    "UBS Investment Bank",
    "UBS Wealth Management",
    "UBS Financial Services Inc.",
    "UBS Financial Services Inc,",
    "UBS Private Wealth Management",
    "UBS Financial Services",
    "UBS AG",
    "UBS Global Asset Management",
    "UBS Global Asset Management - Global Real Estate",
    "UBS Global Wealth Management",
    "UBS Hedge Fund Solutions",
    "UBS OCONNOR LLC",
    "UBS O’Connor",
    "UBS on behalf of STS Structured Products",
    "UBS/Credit Suisse"
}

# List of all variations of Citi names
CITI_VARIATIONS = {
    "Citi",
    "Citigroup",
    "Citibank",
    "Citi Private Bank",
    "Citi Group",
    "Citi Investment Managment",
    "Citibank India",
    "Citigroup Global Markets",
    "Citigroup Inc. (Citi)"
}

# List of all variations of McKinsey names
MCKINSEY_VARIATIONS = {
    "McKinsey & Company",
    "Mckinsey",
    "McKinsey & Co"
}

# List of all variations of EY names
EY_VARIATIONS = {
    "Ernst & Young",
    "EY",
    "ERNST & YOUNG U.S. LLP"
}

# Function to normalize company names
def normalize_company_name(name):
    if name is None:
        return None
        
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and extra spaces
    name = ' '.join(name.split())
    
    # Check if the name matches any of our variation sets
    for variation in JPMORGAN_VARIATIONS:
        if variation.lower() == name:
            return "jp morgan"
    
    for variation in GOLDMAN_SACHS_VARIATIONS:
        if variation.lower() == name:
            return "goldman sachs"
    
    for variation in BANK_OF_AMERICA_VARIATIONS:
        if variation.lower() == name:
            return "bank of america"
    
    for variation in MORGAN_STANLEY_VARIATIONS:
        if variation.lower() == name:
            return "morgan stanley"
    
    for variation in UBS_VARIATIONS:
        if variation.lower() == name:
            return "ubs"
    
    for variation in CITI_VARIATIONS:
        if variation.lower() == name:
            return "citi"
    
    for variation in MCKINSEY_VARIATIONS:
        if variation.lower() == name:
            return "mckinsey"
    
    for variation in EY_VARIATIONS:
        if variation.lower() == name:
            return "ey"
    
    # If we get here, return the original name unchanged
    return name

# Read the input file
with open('summarized_profiles_with_normalized.json', 'r') as f:
    data = json.load(f)

# Process each profile
for profile in data:
    # If we have both current_company_name and normalized_company_name
    if 'current_company_name' in profile and 'normalized_company_name' in profile:
        # Get the standardized name based on current_company_name
        standardized_name = normalize_company_name(profile['current_company_name'])
        # Only update normalized_company_name if we found a match
        if standardized_name != profile['current_company_name']:
            profile['normalized_company_name'] = standardized_name

# Write the updated data back to a new file
with open('summarized_profiles_with_normalized_companies.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Processing complete! Check the new file: summarized_profiles_with_normalized_companies.json")

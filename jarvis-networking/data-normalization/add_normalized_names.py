import pandas as pd
import json
import ijson  # for streaming JSON processing
from tqdm import tqdm
import os

def create_company_mapping():
    """Create a mapping of original company names to normalized names"""
    print("Loading normalized company names...")
    df = pd.read_csv('normalized_company_names.csv')
    
    # Create a dictionary mapping original names to normalized names
    company_mapping = {}
    
    # Add both original and normalized names to the mapping
    for _, row in df.iterrows():
        original_name = str(row['original_name']).lower().strip()
        normalized_name = str(row['normalized_name']).lower().strip()
        company_mapping[original_name] = normalized_name
    
    print(f"Created mapping with {len(company_mapping)} company names")
    return company_mapping

def process_profiles(input_json, output_json, company_mapping):
    """Process profiles and add normalized company names"""
    print("Processing profiles...")
    
    # Get file size for progress bar
    file_size = os.path.getsize(input_json)
    
    # Process the JSON file in chunks
    with open(input_json, 'rb') as infile, open(output_json, 'w') as outfile:
        # Write the opening bracket
        outfile.write('[\n')
        
        # Create a parser for the JSON array
        profiles = ijson.items(infile, 'item')
        
        # Track if we're processing the first item
        first_item = True
        
        # Process each profile
        for profile in tqdm(profiles):
            if not first_item:
                outfile.write(',\n')
            first_item = False
            
            # Get the current company name, handle None values
            current_company = profile.get('current_company_name', '')
            if current_company:
                current_company = current_company.lower().strip()
                
                # Look up the normalized name
                normalized_name = company_mapping.get(current_company, current_company)
            else:
                normalized_name = ''
            
            # Add the normalized name to the profile
            profile['normalized_company_name'] = normalized_name
            
            # Write the profile to the output file
            json.dump(profile, outfile)
        
        # Write the closing bracket
        outfile.write('\n]')

def main():
    input_json = 'summarized_profiles.json'
    output_json = 'summarized_profiles_with_normalized.json'
    
    # Create the company name mapping
    company_mapping = create_company_mapping()
    
    # Process the profiles
    process_profiles(input_json, output_json, company_mapping)
    
    print("Done! Output written to:", output_json)

if __name__ == "__main__":
    main() 
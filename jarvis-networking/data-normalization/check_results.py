import json
import random

def sample_profiles():
    # Load a few samples from both files
    with open('summarized_profiles.json', 'r') as f:
        original_data = json.load(f)
    
    with open('summarized_profiles_with_normalized.json', 'r') as f:
        normalized_data = json.load(f)
    
    # Sample 5 random indices
    sample_size = 5
    total_profiles = len(original_data)
    indices = random.sample(range(total_profiles), sample_size)
    
    print(f"\nChecking {sample_size} random profiles:\n")
    
    for idx in indices:
        orig_profile = original_data[idx]
        norm_profile = normalized_data[idx]
        
        print("Profile", idx)
        print("Original company name:", orig_profile.get('current_company_name', 'N/A'))
        print("Normalized company name:", norm_profile.get('normalized_company_name', 'N/A'))
        print("-" * 80)

if __name__ == "__main__":
    sample_profiles() 
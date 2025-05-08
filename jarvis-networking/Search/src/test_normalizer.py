import json
from data_standardizer import DataStandardizer

def test_normalization():
    """Test normalize_profile with cleaned_profiles.json"""
    standardizer = DataStandardizer()
    
    with open('processed/cleaned_profiles.json') as f:
        profiles = json.load(f)
    
    for i, profile in enumerate(profiles[:5]):  # Test first 5 profiles
        print(f"\n=== Testing Profile {i} ===")
        try:
            normalized = standardizer.normalize_profile(profile)
            print("Successfully normalized:")
            print(json.dumps(normalized, indent=2))
        except Exception as e:
            print(f"Normalization failed: {str(e)}")
            print("Original profile:")
            print(json.dumps(profile, indent=2))

if __name__ == "__main__":
    test_normalization()
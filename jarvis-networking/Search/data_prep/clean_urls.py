import json
import logging
import re
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional, List, Dict

def clean_url(url):
    """Clean and normalize a single URL"""
    if not url or not isinstance(url, str):
        return url
    
    # Remove ALL parameters and fragments first
    cleaned_url = re.sub(r'[?#].*$', '', url)
    
    # Preserve exact path for root profile URLs
    if 'linkedin.com/in/' in cleaned_url and not cleaned_url.endswith('/in/[normalized]'):
        return cleaned_url
    
    # Normalize all other LinkedIn URLs
    normalized_url = re.sub(
        r'(https?://[a-z0-9.-]*linkedin\.com/(?:in|school|company)/)[^/]+',
        r'\1[normalized]',
        cleaned_url,
        flags=re.IGNORECASE
    )
    
    return normalized_url

def sanitize_url(url):
    return clean_url(url)

def clean_profile_urls(input_json_path: str, output_json_path: Optional[str] = None) -> List[Dict]:
    """Clean profile URLs while preserving only the main profile URL"""
    URL_FIELDS_TO_REMOVE = {
        'link', 'url_text', 'profile_url',  # Alternate URL fields
        'company_url', 'institute_url',     # Organization URLs
        'logo_url', 'image_url'             # Media URLs
    }
    
    def clean_urls(obj, is_root_profile=False):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if k in URL_FIELDS_TO_REMOVE:
                    continue
                elif k == 'url':
                    # Only keep if it's the root profile URL
                    cleaned[k] = v if is_root_profile else None
                else:
                    cleaned[k] = clean_urls(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_urls(item) for item in obj]
        return obj

    try:
        with open(input_json_path) as f:
            data = json.load(f)
            profiles = data.get('data', data) if isinstance(data, dict) else data
            
        cleaned_profiles = [clean_urls(profile, is_root_profile=True) 
                          for profile in profiles if isinstance(profile, dict)]
        
        if output_json_path:
            Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(cleaned_profiles, f, indent=2)
        
        return cleaned_profiles
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_urls.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    clean_profile_urls(input_path, output_path)
    print(f"Cleaned data saved to {output_path}")
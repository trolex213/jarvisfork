#!/usr/bin/env python3
"""
filter_db.py - Filter JSON data using natural language queries with local Ollama

Usage:
  python filter_db.py --input input.json --output filtered.json --query "your query"
"""

import json
import argparse
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
import sys
import orjson
from concurrent.futures import ThreadPoolExecutor
import os
import time

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate JSON data"""
    with open(file_path) as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON data should be an array of objects")
        return data

class FilterGenerator:
    def __init__(self):
        self.llm = OllamaLLM(
            model="gemma3:12b",
            temperature=0.05,
            num_ctx=4096
        )

    def generate_filter(self, data_sample: List[Dict], query: str) -> str:
        """Map query to one valid category"""
        valid_categories = [
            'Health Care', 'Financial Services', 'Professional Services',
            'Media and Entertainment', 'Real Estate', 'Lending and Investments',
            'Gaming', 'Manufacturing', 'Other', 'Blockchain & Cryptocurrency',
            'Education', 'Advertising', 'Energy', 'Music and Audio',
            'Venture Capital', 'Artificial Intelligence', 'Platforms',
            'Science and Engineering', 'Design', 'Private Equity',
            'Sales and Marketing', 'Content and Publishing', 'Information Technology',
            'Publishing', 'Community and Lifestyle', 'Agriculture and Farming',
            'Commerce and Shopping', 'Payments', 'Cybersecurity', 'Unknown',
            'Consumer Goods', 'Internet Services', 'Clothing and Apparel',
            'Apps', 'Sports', 'Administrative Services', 'Software',
            'Data and Analytics', 'Travel and Tourism', 'Transportation',
            'Mobile', 'Hardware', 'Public Relations', 'Privacy and Security',
            'Biotechnology', 'Security and Privacy', 'Consumer Electronics',
            'Video', 'Sustainability', 'Government and Military',
            'Construction', 'Events', 'Insurance', 'Natural Resources',
            'Food and Beverage', 'Governance', 'Social Media', 'Philanthropy',
            'Consulting', 'Messaging and Telecommunications', 'Chemicals',
            'Arts and Entertainment', 'Medical Device', 'Engineering',
            'Navigation and Mapping', 'Pharmaceuticals',
            'Entrepreneurship Lab (ELabNYC) Bio & Health Tech', 'Restaurants',
            'Engineering and Science', 'Arts and Commerce', 'Investment Banking',
            'Retail and Shopping', 'Restoration Services', 'Nonprofit',
            'Photography', 'Telecommunications', 'Lifestyle', 'Entertainment',
            'Lighting', 'Human Resources', 'Architecture', 'Hospitality',
            'Venture Capital Firm', 'Cloud Computing', 'Accounting', 'Utilities',
            'Logistics and Transportation', 'Public Relations and Marketing',
            'Home and Construction', 'Games', 'Fitness', 'Beverage',
            'Customer Experience', 'Marketing and Sales', 'Pharmaceutical Manufacturing',
            'Entertainment and Media', 'Retail', 'Engineering and Construction',
            'Non-profit', 'Arts and Culture', 'Legal Services', 'Water and Sustainability',
            'Lenture Capital', 'Marketing', 'Security', 'Technology',
            'Environmental Services', 'Pharmaceutical', 'Automotive', 'Think Tank',
            'Arts and Crafts', 'Home Improvement', 'Robotics', 'Creative Industries',
            'Supply Chain', 'Pharmaceutical Services', 'Home and Living',
            'Investment Management', 'Market Research', 'Marketing and Advertising',
            'Water Treatment', 'Private Family Office', 'Facilities'
        ]
        
        prompt = f"""
        [TASK] Select ONE category that best matches this query
        [AVAILABLE CATEGORIES]
        {', '.join(valid_categories)}
        
        [RULES]
        1. Choose ONLY ONE category from the list
        2. Prioritize the most specific match
        3. Return ONLY the category name
        
        [QUERY]
        {query}
        
        [EXAMPLE 1]
        Query: "AI researchers"
        Output: "Artificial Intelligence"
        
        [EXAMPLE 2]
        Query: "investment bankers"
        Output: "Investment Banking"
        """
        
        response = self.llm.invoke(prompt).strip()
        return response if response in valid_categories else "Unknown"

def apply_filter(data: List[Dict], filter_code: str) -> List[Dict]:
    """Apply the generated filter"""
    if not filter_code.startswith('lambda item:'):
        raise ValueError("Invalid filter format")
    
    filter_func = eval(filter_code)
    return [item for item in data if filter_func(item)]

def fast_category_filter(input_file: str, output_file: str, category: str):
    """
    Filters profiles by category from input JSON file and writes matches to output file.
    Handles both dictionary and list input formats.
    """
    with open(input_file, 'rb') as f:
        data = orjson.loads(f.read())
    
    # Try to find the actual profiles data - could be under 'profiles' key or be the root
    if isinstance(data, dict) and 'profiles' in data:
        items = data['profiles']
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Could not find profile data in JSON structure")
    
    # # Debug: Print actual profile structure if found
    # if items and len(items) > 0 and isinstance(items[0], dict):
    #     print("\n=== First Actual Profile Structure ===")
    #     print(json.dumps(items[0], indent=2, default=str))
    
    matches = []
    for item in items:
        # Check for categories in different possible locations
        profile_categories = []
        if isinstance(item, dict):
            # Look for both 'categories' and 'Category' fields
            if 'categories' in item:
                profile_categories = item['categories']
            elif 'Category' in item:
                profile_categories = item['Category']
            
            # Handle case where category might be a string instead of list
            if isinstance(profile_categories, str):
                profile_categories = [profile_categories]
            
            # Check if our category matches
            if category in profile_categories:
                matches.append(item)
    
    # Write matches to output file
    with open(output_file, 'wb') as f:
        f.write(orjson.dumps(matches, option=orjson.OPT_INDENT_2))

def main():
    parser = argparse.ArgumentParser(description='Filter JSON by query')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--query', required=True, help='Filter query to map to category')
    args = parser.parse_args()
    
    try:
        # Generate category from query
        category = FilterGenerator().generate_filter([], args.query)
        print(f"Query mapped to category: {category}")
        
        if args.input.endswith('.json'):
            fast_category_filter(args.input, args.output, category)
            print(f"Filtered data saved to {args.output}")
    except Exception as e:
        print(f"Error: {e}")

def test_generate_filter():
    """Test the filter generation and application"""
    
    print("=== Testing Query → Category Mapping ===")
    test_queries = [
        "I am looking to match with professors and startups in the computer science area, with a specific focus on AI and Machine Learning Optimization if possible",
        "People in industries/roles such as Finance, Fintech, Wealth Management, Data Science, and Data Analytics.",
        "I go to the University of Waterloo and I'm looking for SWE (software engineering) jobs right now",
        "I'm looking to connect with folks from the chip design industry so basically those who are in front end RTL design, digital verification and computer architecture.",
        "I am in NYU Stern right now, and I am looing to connect people in finance area, such as. investment banking, private equity, buy sides, and consulting!",
        "Most looking for IB/buyside people to connect with for full time recruiting.",
        "I'm interest in entry level or junior roles in cybersecurity and network engineer",
        "I'm looking for position as Data engineer and data scientist and would love to interact with right people hiring.",
        "I am currently studying Computer Science at the University of Massachusetts Amherst and seeking opportunities in data science or software engineering.",
        "I am looking for 2026 summer SWE/ML/Quant internships.",
        "Data Analyst, Data Scientist Roles."
    ]
    
    # Use the correct processed data path
    test_input = "processed/normalized_categoried_data.json"
    
    # Verify file exists
    if not os.path.exists(test_input):
        print(f"ERROR: Input file not found at {test_input}")
        return
    
    # Test query mapping and filtering for each query
    for query in test_queries:
        start_time = time.time()
        category = FilterGenerator().generate_filter([], query)
        elapsed = time.time() - start_time
        
        print(f"Query: '{query[:50]}...'")
        print(f"→ Category: {category}")
        print(f"LLM processing time: {elapsed:.2f}s")
        
        # Define test output path for this category
        test_output = f"test_{category.lower().replace(' ', '_')}.json"
        
        try:
            start_time = time.time()
            fast_category_filter(test_input, test_output, category)
            elapsed = time.time() - start_time
            
            with open(test_output, 'rb') as f:
                filtered = orjson.loads(f.read())
                
            print(f"Filtered {len(filtered)} profiles in {elapsed:.2f}s")
            
            # Cleanup
            os.remove(test_output)
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    # Run either main or tests
    if "--test" in sys.argv:
        test_generate_filter()
    else:
        main()

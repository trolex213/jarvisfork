import json
from collections import defaultdict

def analyze_matches(input_file):
    # Initialize counters
    match_counts = defaultdict(int)
    total_lines = 0
    valid_json = 0
    found_matches = 0
    type_counts = defaultdict(int)
    top_institutions = []  # Will store (institution_name, match_count) tuples
    
    # Process file line by line to handle large files
    with open(input_file, 'r') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                valid_json += 1
                data_type = type(data).__name__
                type_counts[data_type] += 1
                
                # For integer values, count them directly as matches
                if isinstance(data, int):
                    found_matches += 1
                    match_counts[data] += 1
                
                # Only process dictionary entries (skip integers)
                if isinstance(data, dict):
                    for institution, details in data.items():
                        if 'match_count' in details:
                            # Add to our top institutions list
                            top_institutions.append((institution, details['match_count']))
                            
                            # Keep only top 10 sorted by match_count
                            top_institutions.sort(key=lambda x: -x[1])
                            if len(top_institutions) > 10:
                                top_institutions = top_institutions[:10]
                
            except json.JSONDecodeError:
                continue
    
    # Print summary statistics
    print(f"Processed {total_lines} total lines")
    print(f"{valid_json} lines contained valid JSON")
    print(f"JSON type counts: {dict(type_counts)}")
    print(f"{found_matches} lines contained matches")
    
    # Print top 10 match counts
    if match_counts:
        print("Top 10 Match Counts (sorted by frequency):")
        for count, freq in sorted(match_counts.items(), key=lambda x: (-x[1], x[0]))[:10]:
            print(f"{count}: {freq} occurrences")
    
    # Print top 10 institutions by match_count
    if top_institutions:
        print("\nTop 10 Institutions by Match Count:")
        for idx, (institution, count) in enumerate(top_institutions, 1):
            print(f"{idx}. {institution}: {count} matches")

def sort_profiles_by_matches(profiles_data):
    """Sort profiles by total number of matches across all items"""
    return sorted(
        profiles_data,
        key=lambda x: sum(match['match_count'] for match in x['matches'].values()),
        reverse=True
    )

def count_matches_per_profile(profiles_data):
    """Count total matches for each profile"""
    return [
        {
            'profile': profile['profile'].split('\n')[0],  # Get first line (name)
            'total_matches': sum(match['match_count'] for match in profile['matches'].values())
        }
        for profile in profiles_data
    ]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_matches.py <matches_file.json>")
        sys.exit(1)
    
    analyze_matches(sys.argv[1])

import argparse
import json
from typing import List, Dict, Any
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from fuzzywuzzy import fuzz
from collections import defaultdict

class ProfileMatcher:
    def __init__(self, profile_text: str, patterns: Dict[str, Any] = None):
        self.profile_text = profile_text.lower()
        self.patterns = patterns if patterns else {"patterns": []}
    
    def generate_regex_matchers(self, connection_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate more flexible patterns that handle common variations"""
        if self.patterns["patterns"]:  # Check if we already have patterns
            return self.patterns
        
        print("\n=== DEBUG: Generating regex matchers ===")
        print(f"Input connection points: {json.dumps(connection_points[:3], indent=2)}...")
        
        llm = OllamaLLM(
            model="gemma3:12b",
            temperature=0.05,
            num_ctx=4096
        )
        
        prompt = PromptTemplate(
            template="""Generate JSON with regex patterns for these connection points:
            {connection_points_json}
            
            Return ONLY valid JSON in this exact format:
            {{
              "patterns": [
                {{"item": "...", "pattern": "..."}},
                ...
              ]
            }}
            
            Include:
            - Word boundaries (\\b)
            - Common abbreviations
            - Case insensitive matching
            """,
            input_variables=["connection_points_json"]
        )
        
        chain = prompt | llm | JsonOutputParser()
        
        try:
            connection_points_json = json.dumps(connection_points, indent=2)
            raw_output = llm.invoke(prompt.format(connection_points_json=connection_points_json))
            
            # Extract JSON from markdown or raw text
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM output")
                
            json_str = raw_output[json_start:json_end]
            
            # Validate JSON structure
            patterns = json.loads(json_str)['patterns']
            if not isinstance(patterns, list) or \
               not all(isinstance(p, dict) and 'item' in p and 'pattern' in p for p in patterns):
                raise ValueError("Invalid patterns format")
                
            self.patterns = {"patterns": patterns}  # Store patterns in instance
            return self.patterns
            
        except Exception as e:
            print(f"Error generating patterns: {str(e)}\nRaw output: {raw_output[:200]}")
            return {"patterns": []}

    def match_with_generated_patterns(self, connection_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Match using pre-generated patterns"""
        results = defaultdict(list)
        
        for pattern_item in self.patterns.get("patterns", []):
            pattern = pattern_item["pattern"]
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, point in enumerate(connection_points):
                    # Check all relevant text fields
                    texts = [
                        point.get("description", ""),
                        point.get("item", ""),
                        point.get("reason", "")
                    ]
                    for text in texts:
                        text = text.lower()
                        matches = list(regex.finditer(text))
                        if matches:
                            for match in matches:
                                start, end = match.span()
                                context = self._get_context(text, start, end)
                                results[pattern_item["item"]].append({
                                    "matched_text": match.group(),
                                    "context": context,
                                    "position": i
                                })
                                # print(f"Found match for {pattern_item['item']}: {match.group()}")
            except re.error as e:
                print(f"Invalid regex pattern '{pattern}': {str(e)}")
        
        # Sort results by match count
        sorted_results = dict(sorted(
            results.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        ))
        
        return sorted_results

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding text context for a match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        # Add ellipsis if not at start/end of text
        prefix = '...' if context_start > 0 else ''
        suffix = '...' if context_end < len(text) else ''
        
        return f"{prefix}{text[context_start:context_end]}{suffix}"

    def match_connections(self, target_item, connection_points):
        """
        Match a target item against connection points and return sorted matches.
        
        Args:
            target_item (str): The item to match against (e.g. "NYU")
            connection_points (list): List of connection point dicts from the JSON
            
        Returns:
            list: Sorted list of matches with scores, highest first
        """
        matches = []
        
        for point in connection_points:
            # Basic string matching on the item name
            item_similarity = fuzz.ratio(target_item.lower(), point["item"].lower()) / 100
            
            # Combine with connection strength (normalized)
            combined_score = (item_similarity * 0.6) + (point["connection_strength"] * 0.4 / 10)
            
            matches.append({
                "item": point["item"],
                "reason": point["reason"],
                "connection_strength": point["connection_strength"],
                "connection_potential": point["connection_potential"],
                "match_score": combined_score,
                "original_data": point
            })
        
        # Sort by match score descending
        return sorted(matches, key=lambda x: x["match_score"], reverse=True)

def extract_connection_points(json_path: str) -> List[Dict[str, Any]]:
    """Extract connection points from the nested JSON structure"""
    with open(json_path) as f:
        data = json.load(f)
    return data['connections']['connection_points']

def extract_profile_texts(json_path: str) -> List[str]:
    """Extract all relevant text fields from each profile in the JSON"""
    with open(json_path) as f:
        profiles = json.load(f)
    
    profile_texts = []
    for profile in profiles:
        text_parts = []
        
        # Include embedding_text first if it exists
        if profile.get('embedding_text'):
            text_parts.append(profile['embedding_text'])
            
        # Add basic fields
        for field in ['about', 'current_company_name', 'position', 'location', 'name']:
            if profile.get(field):
                text_parts.append(str(profile[field]))
                
        # Handle education (list of dicts)
        if profile.get('education'):
            for edu in profile['education']:
                text_parts.append(edu.get('description', ''))
                text_parts.append(edu.get('school', ''))
                text_parts.append(edu.get('degree', ''))
                
        # Handle experience (list of dicts)
        if profile.get('experience'):
            for exp in profile['experience']:
                text_parts.append(exp.get('description', ''))
                text_parts.append(exp.get('company', ''))
                text_parts.append(exp.get('position', ''))
                
        # Handle other list fields
        for field in ['projects', 'honors_and_awards', 'volunteer_experience']:
            if profile.get(field):
                for item in profile[field]:
                    text_parts.append(str(item))
                    
        # Join all parts with newlines
        profile_texts.append('\n'.join(filter(None, text_parts)))
    
    return profile_texts

def generate_matching_matrix(all_results: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Create a dictionary showing which profiles matched with which items
    
    Returns:
        Dict where keys are item names and values are lists of profile indices that matched
    """
    matching_matrix = defaultdict(list)
    
    for profile_idx, result in enumerate(all_results):
        for item_name, match_data in result['matches'].items():
            if match_data['match_count'] > 0:
                matching_matrix[item_name].append(profile_idx)
    
    return matching_matrix

def get_profile_matches_list(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process raw match results into final output format"""
    profile_matches = []
    
    for result in all_results:
        if isinstance(result, dict) and result.get('matches'):
            profile_matches.append({
                'profile': result.get('profile', {}),
                'matches': result['matches'],
                'total_score': result.get('total_score', 0),
                'match_count': len(result['matches'])
            })
    
    return profile_matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Match profiles against connection points')
    parser.add_argument('--profiles', type=str, required=True,
                       help='JSON file containing array of profile objects with embedding_text')
    parser.add_argument('--connection_points', type=str, required=True,
                       help='JSON file containing connection points in nested structure')
    parser.add_argument('--output', type=str, default='matches.json',
                       help='Output file for results (default: matches.json)')
    
    args = parser.parse_args()
    
    # Load input files
    profile_texts = extract_profile_texts(args.profiles)
    connection_points = extract_connection_points(args.connection_points)
    
    # Process each profile
    all_results = []
    matcher = ProfileMatcher(profile_texts[0], None)
    matcher.generate_regex_matchers(connection_points)
    for idx, profile_text in enumerate(profile_texts):
        matcher = ProfileMatcher(profile_text, matcher.patterns)
        results = matcher.match_with_generated_patterns(connection_points)
        all_results.append({
            'profile': profile_text[:100] + '...',  # Store preview
            'matches': results
        })
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Processed {len(profile_texts)} profiles. Results saved to {args.output}")
    
    # matching_matrix = generate_matching_matrix(all_results)
    # print("\nMatching Matrix (items -> profile indices):")
    # for item, profiles in matching_matrix.items():
    #     print(f"{item}: {profiles}")
    
    profile_matches = get_profile_matches_list(all_results)
    # Sort profiles by number of matches (descending) and take top 10
    sorted_profiles = sorted(
        enumerate(profile_matches),
        key=lambda x: x[1]['match_count'],
        reverse=True
    )[:10]
    print("\nTop 10 Profiles by Number of Matches:")
    for idx, matches in sorted_profiles:
        print(f"\n=== Profile {idx} ===")
        print(f"Matches ({matches['match_count']}):")
        for item, match_data in matches['matches'].items():
            print(f"  {item}:")
            for i, m in enumerate(match_data[:3]):  # Show first 3 matches per item
                print(f"    Match {i+1}: '{m['matched_text']}' in context: '...{m['context']}...'")
    
    # Get top 10 matches by count
    # top_matches = sorted(
    #     matching_matrix.items(),
    #     key=lambda x: len(x[1]),
    #     reverse=True
    # )[:10]
    
    # print("\nTop 10 Matched Items:")
    # for item, profiles in top_matches:
    #     print(f"{item}: {len(profiles)} profiles")

import argparse
import json
from typing import List, Dict, Any, Union, Optional
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from fuzzywuzzy import fuzz
from collections import defaultdict
import sys

class ProfileMatcher:
    def __init__(self):
        self.profile_text = ""
        self.patterns = {"patterns": []}
        self._cached_patterns = None
        self._cached_connection_points = None
        
    def set_profile_text(self, text):
        """Set the profile text without recreating the matcher"""
        self.profile_text = text.lower()
    
    def generate_regex_matchers(self, connection_points):
        """Generate regex patterns with caching"""
        if self._cached_connection_points == connection_points:
            return self.patterns
        
        self._cached_connection_points = connection_points
        print("\n=== DEBUG: Generating regex matchers ===")
        print(f"Input connection points: {json.dumps(connection_points[:3], indent=2)}...")
        
        llm = OllamaLLM(
            model="gemma3:12b",
            temperature=0.1,
            num_ctx=8192
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
            self._cached_patterns = self.patterns
            return self.patterns
            
        except Exception as e:
            print(f"Error generating patterns: {str(e)}\nRaw output: {raw_output[:200]}")
            return {"patterns": []}

    def match_with_generated_patterns(self, connection_points):
        """Match using pre-generated patterns"""
        results = defaultdict(list)
        
        # print(f"\n=== DEBUG: Matching patterns against connection points ===")
        
        for pattern_item in self.patterns.get("patterns", []):
            pattern = pattern_item["pattern"]
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for i, point in enumerate(connection_points):
                    # Get connection strength (default to 5 if not specified)
                    strength = point.get('connection_strength', 5)
                    
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
                            # print(f"Found {len(matches)} matches for pattern '{pattern}' in point {i}")
                            for match in matches:
                                start, end = match.span()
                                context = self._get_context(text, start, end)
                                # Calculate score (0-100) based on connection strength
                                score = min(100, strength * 10)  # Scale 0-10 to 0-100
                                results[pattern_item["item"]].append({
                                    "matched_text": match.group(),
                                    "context": context,
                                    "position": i,
                                    "score": score,
                                    "strength": strength
                                })
            except re.error as e:
                print(f"Invalid regex pattern '{pattern}': {str(e)}", file=sys.stderr)
        
        # Sort results by total score (sum of all match scores)
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: sum(m['score'] for m in x[1]),
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

def extract_connection_points(connections_data: Union[str, Dict]) -> List[Dict[str, Any]]:
    """Extract connection points from nested JSON (accepts file path or direct data)"""
    if isinstance(connections_data, str):
        with open(connections_data) as f:
            data = json.load(f)
    else:
        data = connections_data
    
    if isinstance(data, dict):
        if 'connections' in data and 'connection_points' in data['connections']:
            return data['connections']['connection_points']
        elif 'connection_points' in data:
            return data['connection_points']
    return data  # Assume it's already the connection points list

def extract_profile_texts(profiles_data: Union[str, List[Dict]]) -> List[str]:
    """Extract all relevant text fields from each profile in the JSON"""
    if isinstance(profiles_data, str):
        with open(profiles_data) as f:
            profiles = json.load(f)
    else:
        profiles = profiles_data
    
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

def match_profile_to_connection_points(matcher, profile_text, connection_points, debug=False):
    """Match a profile using a shared matcher instance"""
    matcher.set_profile_text(profile_text)
    if debug:
        print("\n=== DEBUG: First profile only ===")
        print(f"Connection points: {json.dumps(connection_points[:3], indent=2)}...")
    return {
        'profile': profile_text[:100] + '...',
        'matches': matcher.match_with_generated_patterns(connection_points)
    }

def main(profiles_data=None, connections_data=None, args=None):
    """Main function that can accept direct data or CLI args"""
    if args is None and (profiles_data is None or connections_data is None):
        # Fall back to CLI mode
        # print("fall back?")
        parser = argparse.ArgumentParser()
        parser.add_argument('--profiles', type=str, required=True)
        parser.add_argument('--connection_points', type=str, required=True)
        parser.add_argument('--output', type=str, default='matches.json')
        args = parser.parse_args()
        
        # Load from files if in CLI mode
        with open(args.profiles) as f:
            profiles_data = json.load(f)
        with open(args.connection_points) as f:
            connections_data = json.load(f)

    # Ensure connections_data has the right format
    if isinstance(connections_data, dict) and 'connections' not in connections_data:
        connections_data = {'connections': connections_data}

    # Process data
    profile_texts = extract_profile_texts(profiles_data)
    connection_points = extract_connection_points(connections_data)
    
    # Match profiles
    matcher = ProfileMatcher()
    matcher.generate_regex_matchers(connection_points)
    
    all_results = []
    for i, profile in enumerate(profile_texts):
        results = match_profile_to_connection_points(
            matcher,
            profile,
            connection_points,
            debug=(i == 0)  # Debug first profile only
        )
        if results:
            all_results.append(results)
    
    # Sort all results by total_score (highest first)
    all_results.sort(
        key=lambda x: x.get('total_score', 0), 
        reverse=True
    )
    
    # Sort matches within each profile by score
    for result in all_results:
        if 'matches' in result and isinstance(result['matches'], list):
            result['matches'].sort(
                key=lambda x: x.get('match_score', 0),
                reverse=True
            )
    
    # If in CLI mode, save results
    if args and args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f)
    
    # Get and display top 10 profiles by match count
    profile_matches = get_profile_matches_list(all_results)
    # Sort profiles by number of matches (descending) and take top 10
    sorted_profiles = sorted(
        enumerate(profile_matches),
        key=lambda x: x[1]['match_count'],
        reverse=True
    )[:10]
    print("\nTop 10 Profiles by Number of Matches:")
    for idx, matches in sorted_profiles:
        profile = profiles_data[idx]
        print(f"\n=== Profile {idx} (Name: {profile.get('name', 'N/A')}, URL: {profile.get('url', 'N/A')}) ===")
        print(f"Matches ({matches['match_count']}):")
        for item, match_data in matches['matches'].items():
            print(f"  {item} (Total Score: {sum(m['score'] for m in match_data)}):")
            for i, m in enumerate(match_data[:1]):  # Show first 1 matches per item
                print(f"    Match {i+1}: Score={m['score']} Strength={m['strength']} '{m['matched_text']}' in context: '...{m['context']}...'")
    
    return all_results

if __name__ == "__main__":
    main()

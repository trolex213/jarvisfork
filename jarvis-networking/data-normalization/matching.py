import json
import re
from collections import defaultdict
import numpy as np
from difflib import SequenceMatcher

def load_data(resume_path, profiles_path):
    """Load resume and LinkedIn profiles data"""
    with open(resume_path, 'r') as file:
        resume_data = json.load(file)
    
    with open(profiles_path, 'r') as file:
        profiles_data = json.load(file)
    
    return resume_data, profiles_data

def text_similarity(text1, text2):
    """Calculate text similarity between two strings"""
    if not text1 or not text2:
        return 0
    
    # Convert to lowercase and split into words
    text1 = re.sub(r'[^\w\s]', '', str(text1).lower())
    text2 = re.sub(r'[^\w\s]', '', str(text2).lower())
    
    # Use SequenceMatcher for better text similarity
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_education_similarity(resume_edu, profile_edu):
    """Calculate similarity based on education"""
    if not resume_edu or not profile_edu:
        return 0
    
    max_similarity = 0
    
    for r_edu in resume_edu:
        r_institution = r_edu.get('institution', '')
        
        for p_edu in profile_edu:
            p_institution = p_edu.get('title', '')
            
            # Check for institution match
            inst_similarity = text_similarity(r_institution, p_institution)
            
            # Check for degree match if available
            degree_similarity = 0
            if 'degree' in r_edu and 'degree' in p_edu:
                degree_similarity = text_similarity(r_edu['degree'], p_edu['degree'])
            
            # Calculate weighted similarity
            similarity = 0.7 * inst_similarity + 0.3 * degree_similarity
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def calculate_experience_similarity(resume_exp, profile_exp):
    """Calculate similarity based on work experience"""
    if not resume_exp or not profile_exp:
        return 0
    
    similarities = []
    
    for r_exp in resume_exp:
        r_company = r_exp.get('company', '')
        r_position = r_exp.get('position', '')
        r_description = r_exp.get('description', '')
        
        for p_exp in profile_exp:
            p_company = p_exp.get('company', '')
            
            # Get position from profile experience if available
            p_position = ""
            if 'positions' in p_exp and p_exp['positions']:
                p_position = p_exp['positions'][0].get('title', '')
            
            # Calculate similarity for company and position
            company_sim = text_similarity(r_company, p_company)
            position_sim = text_similarity(r_position, p_position)
            
            # Calculate weighted similarity
            similarity = 0.6 * company_sim + 0.4 * position_sim
            similarities.append(similarity)
    
    # Return the average of top 2 similarities or 0 if none found
    similarities.sort(reverse=True)
    if similarities:
        return sum(similarities[:min(2, len(similarities))]) / min(2, len(similarities))
    return 0

def calculate_keyword_similarity(resume_keywords, profile_data):
    """Calculate similarity based on professional keywords"""
    if not resume_keywords:
        return 0
    
    # Extract text from profile
    profile_text = json.dumps(profile_data).lower()
    
    # Split keywords and count matches
    keywords = resume_keywords.lower().split(', ')
    matches = sum(1 for keyword in keywords if keyword.lower() in profile_text)
    
    return matches / len(keywords) if keywords else 0

def calculate_location_similarity(resume_location, profile_location):
    """Calculate similarity based on location"""
    if not resume_location or not profile_location:
        return 0
    
    # Simple location matching (could be enhanced with geographic proximity)
    resume_location = resume_location.lower()
    profile_location = profile_location.lower()
    
    # Check if city or state matches
    if resume_location in profile_location or profile_location in resume_location:
        return 1.0
    
    return text_similarity(resume_location, profile_location)

def find_top_matches(resume_data, profiles_data, top_k=5):
    """Find top k matches for a resume in the profiles database"""
    resume = resume_data['resume_data']
    connections = resume_data['connections']['connection_points']
    
    # Extract connection items for matching
    connection_items = {conn['item'].lower(): conn['connection_strength'] for conn in connections}
    
    results = []
    
    for profile in profiles_data:
        score = 0
        weights = {}
        
        # Calculate education similarity
        edu_similarity = calculate_education_similarity(resume.get('education', []), profile.get('education', []))
        weights['education'] = 0.35  # Higher weight for education
        score += edu_similarity * weights['education']
        
        # Calculate experience similarity
        exp_similarity = calculate_experience_similarity(resume.get('experience', []), profile.get('experience', []))
        weights['experience'] = 0.25
        score += exp_similarity * weights['experience']
        
        # Calculate keyword similarity
        keyword_similarity = calculate_keyword_similarity(resume.get('professional_keywords', ''), profile)
        weights['keywords'] = 0.15
        score += keyword_similarity * weights['keywords']
        
        # Calculate location similarity
        loc_similarity = calculate_location_similarity(resume.get('city', ''), profile.get('city', ''))
        weights['location'] = 0.10
        score += loc_similarity * weights['location']
        
        # Calculate connection point matches
        connection_score = 0
        profile_text = json.dumps(profile).lower()
        
        for item, strength in connection_items.items():
            if any(word in profile_text for word in item.lower().split()):
                # Normalize connection strength to 0-1 scale
                connection_score += (strength / 10)
        
        # Normalize connection score
        if connection_items:
            connection_score = connection_score / len(connection_items)
            weights['connections'] = 0.15
            score += connection_score * weights['connections']
        
        # Store the match with its score
        match_details = {
            'name': profile.get('name', 'Unknown'),
            'position': profile.get('position', 'Unknown'),
            'company': profile.get('current_company_name', 'Unknown'),
            'education': profile.get('educations_details', 'Unknown'),
            'location': profile.get('city', 'Unknown'),
            'linkedin_url': profile.get('url', 'Unknown'),  # Add LinkedIn URL
            'score': round(score, 2),
            'match_reasons': {
                'education': round(edu_similarity, 2),
                'experience': round(exp_similarity, 2),
                'keywords': round(keyword_similarity, 2),
                'location': round(loc_similarity, 2),
                'connections': round(connection_score, 2) if connection_items else 0
            }
        }
        
        results.append(match_details)
    
    # Sort by score and return top k matches
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def main():
    # Example usage
    resume_path = "charles_hu_resume.json"
    profiles_path = "summarized_profiles.json"
    
    resume_data, profiles_data = load_data(resume_path, profiles_path)
    top_matches = find_top_matches(resume_data, profiles_data, top_k=5)
    
    print(f"Top matches for {resume_data['resume_data']['name']}:")
    for i, match in enumerate(top_matches, 1):
        print(f"\n{i}. {match['name']} - {match['position']}")
        print(f"   Company: {match['company']}")
        print(f"   Education: {match['education']}")
        print(f"   Location: {match['location']}")
        print(f"   LinkedIn: {match['linkedin_url']}")  # Add LinkedIn URL to output
        print(f"   Match score: {match['score']}")
        print("   Match reasons:")
        for reason, score in match['match_reasons'].items():
            print(f"      - {reason.capitalize()}: {score}")

if __name__ == "__main__":
    main()
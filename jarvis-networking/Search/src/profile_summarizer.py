"""
LinkedIn Profile Summarizer

This script enhances LinkedIn profile data by:
1. Generating comprehensive summaries from metadata
2. Preserving original 'about' sections
3. Creating embedding-ready combined text

Usage Examples:

1. File-based processing (large datasets):
   from profile_summarizer import process_dataset
   process_dataset('input.json', 'output.json')

2. In-memory processing (Python integration):
   from profile_summarizer import summarize_profiles
   with open('input.json') as f:
       profiles = json.load(f)['data']
   enhanced = summarize_profiles(profiles)

3. Command-line usage:
   python profile_summarizer.py --input raw_profiles.json --output enhanced_profiles.json
"""
import json
import argparse
from typing import Dict, Any, List
from keybert import KeyBERT
import logging
from tqdm import tqdm
import time
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# Initialize KeyBERT once at module level
kw_model = KeyBERT('all-MiniLM-L6-v2')

def generate_metadata_summary(profile: Dict[str, Any]) -> str:
    """Generate a comprehensive text summary including descriptions"""
    sections = []
    
    # Basic info section
    basic_info = []
    if profile.get("name"):
        basic_info.append(f"Name: {profile['name']}")
    if profile.get("position"):
        basic_info.append(f"Position: {profile['position']}")
    if basic_info:
        sections.append("\n".join(basic_info))
    
    # Experience summary with descriptions
    if profile.get("experience"):
        exp_section = ["Experience:"]
        for exp in profile["experience"][:3]:  # Top 3 positions
            if not exp:
                continue
            exp_parts = []
            if exp.get("title"): exp_parts.append(f"Role: {exp['title']}")
            if exp.get("company"): exp_parts.append(f"Company: {exp['company']}")
            if exp.get("duration"): exp_parts.append(f"Duration: {exp['duration']}")
            if exp_parts:
                exp_section.append(" - " + ", ".join(exp_parts))
                if exp.get("description"):
                    exp_section.append(f"   Description: {exp['description']}")
        sections.append("\n".join(exp_section))
    
    # Education summary with descriptions
    if profile.get("education"):
        edu_section = ["Education:"]
        for edu in profile["education"]:
            if not edu:
                continue
            edu_parts = []
            if edu.get("title"): edu_parts.append(f"Degree: {edu['title']}")
            if edu.get("institute"): edu_parts.append(f"Institute: {edu['institute']}")
            if edu.get("duration"): edu_parts.append(f"Duration: {edu['duration']}")
            if edu_parts:
                edu_section.append(" - " + ", ".join(edu_parts))
                if edu.get("description"):
                    edu_section.append(f"   Description: {edu['description']}")
        sections.append("\n".join(edu_section))
    
    # Add descriptions statistics
    desc_data = extract_descriptions(profile)
    if desc_data['education'] or desc_data['experience']:
        stats = ["Description Insights:"]
        if desc_data['experience']:
            stats.append(f"- {len(desc_data['experience'])} experience descriptions")
        if desc_data['education']:
            stats.append(f"- {len(desc_data['education'])} education descriptions")
        sections.append("\n".join(stats))
    
    return "\n\n".join(sections)


def extract_keywords(text: str, n_keywords: int = 5) -> List[str]:
    """Extract keywords using pre-loaded model"""
    if not text or not text.strip():
        return []
    
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=n_keywords
    )
    return [kw[0] for kw in keywords]


def enhance_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure profile is a dict before copying"""
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except json.JSONDecodeError:
            return None
            
    if not isinstance(profile, dict):
        return None
        
    enhanced = profile.copy()
    
    # Only enhance about section if it exists and has content
    if profile.get('about', '').strip():
        about_text = profile['about']
        
        # Append experience descriptions
        if profile.get('experience'):
            exp_descriptions = [exp.get('description','') for exp in profile['experience'] if exp.get('description')]
            if exp_descriptions:
                about_text += '\n\nProfessional Experience:\n' + '\n'.join(exp_descriptions)
        
        # Append education descriptions
        if profile.get('education'):
            edu_descriptions = [edu.get('description','') for edu in profile['education'] if edu.get('description')]
            if edu_descriptions:
                about_text += '\n\nEducation Background:\n' + '\n'.join(edu_descriptions)
        
        enhanced['about'] = about_text
    
    # Generate summary including all descriptions
    enhanced['generated_summary'] = generate_metadata_summary(profile)
    
    # Benchmark and analyze similarity when both exist
    if profile.get('about', '').strip() and enhanced['generated_summary']:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import time
        
        # Benchmark different models
        models = {
            'miniLM': 'all-MiniLM-L6-v2',
            'mpnet': 'all-mpnet-base-v2'
        }
        
        benchmark_results = {}
        for name, model_name in models.items():
            start = time.time()
            model = SentenceTransformer(model_name)
            about_emb = model.encode(profile['about'])
            summary_emb = model.encode(enhanced['generated_summary'])
            similarity = cosine_similarity([about_emb], [summary_emb])[0][0]
            benchmark_results[name] = {
                'similarity': float(similarity),
                'time_ms': int((time.time()-start)*1000),
                'about_emb': about_emb.tolist(),  # Store full embeddings
                'summary_emb': summary_emb.tolist()
            }
        
        enhanced['similarity_benchmark'] = {
            'profile_id': profile.get('profile_id', hash(profile['about'])),
            'results': benchmark_results,
            'about_length': len(profile['about']),
            'summary_length': len(enhanced['generated_summary'])
        }
    
    # Extract keywords from about + summary
    text = f"{enhanced.get('about', '')}\n{enhanced['generated_summary']}"
    enhanced["keywords"] = extract_keywords(text)
    
    # Extract professional interests from experience/education descriptions
    desc_data = extract_descriptions(profile)
    professional_text = '\n'.join(desc_data['experience'] + desc_data['education'])
    professional_interests = extract_keywords(professional_text, n_keywords=5) if professional_text else []
    
    # Extract personal interests from about section or generated content
    desc_data = extract_descriptions(profile)
    about_text = profile.get('about', enhanced.get('generated_summary', ''))
    
    # Combine with experience/education descriptions
    combined_text = '\n'.join([
        about_text,
        *desc_data['experience'],
        *desc_data['education']
    ])
    
    personal_interests = extract_keywords(combined_text, n_keywords=5) if combined_text.strip() else []
    
    # Add comprehensive interests section
    enhanced["interests"] = {
        "professional": professional_interests,
        "personal": personal_interests,
        "all_keywords": enhanced.get("keywords", [])
    }
    
    # Create combined text for embeddings
    about = profile.get("about")
    about_text = about.strip() if about and isinstance(about, str) else ""
    enhanced["embedding_text"] = (
        f"{about_text}\n\n{enhanced['generated_summary']}" if about_text 
        else enhanced["generated_summary"]
    )
    
    return enhanced


def enhance_profile_batch(profiles: List[Dict]) -> List[Dict]:
    """Batch version of enhance_profile for better performance"""
    if not profiles:
        return []
        
    # Batch process all profiles
    batch_summaries = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
            
        sections = []
        
        # Basic info section
        basic_info = []
        if profile.get("name"):
            basic_info.append(f"Name: {profile['name']}")
        if profile.get("position"):
            basic_info.append(f"Position: {profile['position']}")
        if basic_info:
            sections.append("\n".join(basic_info))
        
        # Experience summary with descriptions
        if profile.get("experience"):
            exp_section = ["Experience:"]
            for exp in profile["experience"][:3]:  # Top 3 positions
                if not exp:
                    continue
                exp_parts = []
                if exp.get("title"): exp_parts.append(f"Role: {exp['title']}")
                if exp.get("company"): exp_parts.append(f"Company: {exp['company']}")
                if exp.get("duration"): exp_parts.append(f"Duration: {exp['duration']}")
                if exp_parts:
                    exp_section.append(" - " + ", ".join(exp_parts))
                    if exp.get("description"):
                        exp_section.append(f"   Description: {exp['description']}")
            sections.append("\n".join(exp_section))
        
        # Education summary with descriptions
        if profile.get("education"):
            edu_section = ["Education:"]
            for edu in profile["education"]:
                if not edu:
                    continue
                edu_parts = []
                if edu.get("title"): edu_parts.append(f"Degree: {edu['title']}")
                if edu.get("institute"): edu_parts.append(f"Institute: {edu['institute']}")
                if edu.get("duration"): edu_parts.append(f"Duration: {edu['duration']}")
                if edu_parts:
                    edu_section.append(" - " + ", ".join(edu_parts))
                    if edu.get("description"):
                        edu_section.append(f"   Description: {edu['description']}")
            sections.append("\n".join(edu_section))
        
        # Add descriptions statistics
        desc_data = extract_descriptions(profile)
        if desc_data['education'] or desc_data['experience']:
            stats = ["Description Insights:"]
            if desc_data['experience']:
                stats.append(f"- {len(desc_data['experience'])} experience descriptions")
            if desc_data['education']:
                stats.append(f"- {len(desc_data['education'])} education descriptions")
            sections.append("\n".join(stats))
        
        batch_summaries.append("\n\n".join(sections))
    
    # Batch generate summaries (assuming model supports batch processing)
    try:
        summaries = model.generate(batch_summaries, max_length=512, num_beams=4)
    except Exception:
        summaries = batch_summaries  # Fallback to raw sections
    
    # Attach summaries to profiles
    return [
        {**profile, "generated_summary": summary, "embedding_text": f"{profile.get('about','')}\n\n{summary}"}
        for profile, summary in zip(profiles, summaries)
        if isinstance(profile, dict)
    ]


def extract_descriptions(profile):
    """Extract education and experience descriptions from profile"""
    education_descriptions = []
    experience_descriptions = []
    
    # Process education descriptions
    if profile.get('education'):
        for edu in profile['education']:
            if edu and isinstance(edu, dict) and edu.get('description'):
                education_descriptions.append(edu['description'])
    
    # Process experience descriptions
    if profile.get('experience'):
        for exp in profile['experience']:
            if exp and isinstance(exp, dict) and exp.get('description'):
                experience_descriptions.append(exp['description'])
    
    return {
        'education': education_descriptions,
        'experience': experience_descriptions
    }


def summarize_profile(profile):
    """Generate a text summary of a LinkedIn profile"""
    sections = []
    
    # 1. Basic Identity
    identity = []
    if profile.get("name"):
        identity.append(f"{profile['name']}")
    if profile.get("position"):
        identity.append(f"{profile['position']}")
    if identity:
        sections.append("\n".join(identity))
    
    # 2. Location Information
    location = []
    if profile.get("city"):
        location.append(profile["city"])
    if profile.get("country_code"):
        location.append(f"({profile['country_code']})")
    if location:
        sections.append(" " + " ".join(location))
    
    # 3. Current Role
    if profile.get("current_company_name"):
        current_role = [f" {profile['current_company_name']}"]
        if isinstance(profile.get("current_company"), dict):
            if profile["current_company"].get("title"):
                current_role.append(f"Role: {profile['current_company']['title']}")
        sections.append("\n".join(current_role))
    
    # 4. Education Summary (now includes descriptions)
    if profile.get("education"):
        edu_lines = []
        for edu in profile["education"]:
            if not edu:
                continue
            parts = []
            if edu.get("title"): parts.append(edu["title"])
            if edu.get("degree"): parts.append(f"| {edu['degree']}")
            if edu.get("field"): parts.append(f"({edu['field']})")
            if edu.get("start_year") and edu.get("end_year"):
                parts.append(f"{edu['start_year']}-{edu['end_year']}")
            if parts:
                edu_line = " ".join(parts)
                if edu.get("description"):
                    edu_line += f"\n  - {edu['description']}"
                edu_lines.append(edu_line)
        if edu_lines:
            sections.append(" Education:\n" + "\n".join(f"- {line}" for line in edu_lines))
    
    # 5. Professional Experience (now includes descriptions)
    if profile.get("experience"):
        exp_lines = []
        for exp in profile["experience"][:3]:  # Top 3 positions
            if not exp:
                continue
            parts = []
            if exp.get("title"): parts.append(exp["title"])
            if exp.get("company"): parts.append(f"@ {exp['company']}")
            if exp.get("duration"): parts.append(f"({exp['duration']})")
            if parts:
                exp_line = " ".join(parts)
                if exp.get("description"):
                    exp_line += f"\n  - {exp['description']}"
                exp_lines.append(exp_line)
        if exp_lines:
            sections.append(" Experience:\n" + "\n".join(f"- {line}" for line in exp_lines))
    
    # 6. Additional Sections
    additional = []
    
    # Honors & Awards
    if profile.get("honors_and_awards"):
        awards = [
            f"{award.get('title', '')} ({award.get('date', '')})" 
            for award in profile["honors_and_awards"] 
            if award.get("title")
        ]
        if awards:
            additional.append(" Honors:\n- " + "\n- ".join(awards))
    
    # Organizations
    if profile.get("organizations"):
        orgs = [org.get("title", "") for org in profile["organizations"] if org.get("title")]
        if orgs:
            additional.append(" Organizations:\n- " + "\n- ".join(orgs))
    
    # Volunteer Work
    if profile.get("volunteer_experience"):
        volunteer = [
            vol.get("title", "") 
            for vol in profile["volunteer_experience"] 
            if vol.get("title")
        ]
        if volunteer:
            additional.append(" Volunteer Work:\n- " + "\n- ".join(volunteer))
    
    if additional:
        sections.append("\n".join(additional))
    
    # 7. Add description statistics
    desc_data = extract_descriptions(profile)
    if desc_data['education'] or desc_data['experience']:
        sections.append("\nDescription Statistics:")
        if desc_data['education']:
            sections.append(f"- Education descriptions: {len(desc_data['education'])}")
        if desc_data['experience']:
            sections.append(f"- Experience descriptions: {len(desc_data['experience'])}")
    
    return "\n\n".join(sections)


def summarize_profiles(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple profiles through enhancement pipeline"""
    return [enhance_profile(profile) for profile in profiles]


def process_dataset(input_path: str, output_path: str, batch_size: int = 64) -> None:
    """Process dataset with true batch support"""
    import json
    from tqdm import tqdm
    
    with open(input_path) as f:
        data = json.load(f)
    
    processed = []
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch = data[i:i+batch_size]
        try:
            enhanced_batch = enhance_profile_batch(batch)
            processed.extend(enhanced_batch)
        except Exception as e:
            # Fallback to individual processing
            processed.extend(p for p in 
                           (enhance_profile(profile) for profile in batch) 
                           if p is not None)
    
    with open(output_path, 'w') as f:
        json.dump(processed, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhance LinkedIn profiles with metadata summaries')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()
    
    # Execute processing
    process_dataset(args.input, args.output)
    print(f" Enhanced dataset saved to {args.output}")

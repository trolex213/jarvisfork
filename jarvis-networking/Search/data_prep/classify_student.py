# classify_student.py
import pandas as pd
import json
import numpy as np
from pathlib import Path
import spacy
import logging

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def get_education_json(idx, data):
    """Flatten education entry with index"""
    try:
        flattened_entry = {
            f'degree_{idx}': data.get('degree'),
            f'field_{idx}': data.get('field'),
            f'title_{idx}': data.get('title'),
            f'description_{idx}': data.get('description'),
            f'end_year_{idx}': data.get('end_year'),
            f'start_year_{idx}': data.get('start_year')
        }
    except:
        flattened_entry = {
            f'degree_{idx}': pd.NA,
            f'field_{idx}': pd.NA,
            f'title_{idx}': pd.NA,
            f'description_{idx}': pd.NA,
            f'end_year_{idx}': pd.NA,
            f'start_year_{idx}': pd.NA
        }
    return flattened_entry

def sort_key(col_name):
    """Sort columns by index then name"""
    parts = col_name.split('_')
    try:
        return (int(parts[-1]), '_'.join(parts[:-1]))  # (number, field_name)
    except ValueError:
        return (-1, col_name)  # Put non-indexed fields first

def edu_json(brightdata):
    """Process education data into flattened DataFrame"""
    education_lists = []
    for row in brightdata['education']:
        if row is not None:
            merged_entry = {}
            for idx, data in enumerate(row):
                flattened_entry = get_education_json(idx, data)
                merged_entry = {**merged_entry, **flattened_entry}
            if merged_entry:
                education_lists.append(merged_entry)
        else:
            education_lists.append({
                f'degree_0': np.nan,
                f'field_0': np.nan,
                f'title_0': np.nan,
                f'description_0': np.nan,
                f'end_year_0': np.nan,
                f'start_year_0': np.nan
            })
    
    education_df = pd.DataFrame(education_lists)
    sorted_cols = sorted(education_df.columns, key=sort_key)
    education_df = education_df[sorted_cols]
    assert len(education_df) == len(brightdata['education'])
    return education_df

def get_current_company_json(data):
    """Flatten company entry"""
    try:
        flattened_entry = {
            'position_title': data.get('title'),
            'company_name': data.get('name')
        }
    except:
        flattened_entry = {
            'position_title': pd.NA,
            'company_name': pd.NA
        }
    return flattened_entry

def current_company_json(brightdata):
    """Process current company data into flattened DataFrame"""
    current_company_lists = []
    for row in brightdata['current_company']:
        if row is not None:
            flattened_entry = get_current_company_json(row)
            current_company_lists.append(flattened_entry)
        else:
            current_company_lists.append({
                'position_title': np.nan,
                'company_name': np.nan
            })
    
    current_company_df = pd.DataFrame(current_company_lists)
    sorted_cols = sorted(current_company_df.columns, key=sort_key)
    current_company_df = current_company_df[sorted_cols]
    assert len(current_company_df) == len(brightdata['current_company'])
    return current_company_df

def analyze_student_status(text, debug=False):
    """Advanced student classification using NLP"""
    if pd.isna(text):
        return False
    
    doc = nlp(str(text).lower())
    
    TERMS = {
        'student': {
            'academic_roles': {'student': 1.0, 'freshman': 1.2, 'sophomore': 1.2, 
                              'junior': 1.2, 'senior': 1.0, 'undergrad': 1.0},
            'academic_verbs': {'study': 0.8, 'major': 1.0, 'enroll': 1.2, 
                              'graduate': 0.5, 'learn': 0.5},
            'academic_orgs': {'university': 0.7, 'college': 0.7, 'school': 0.5}
        },
        'professional': {
            'executive_titles': {'cfo': 2.0, 'ceo': 2.0, 'cto': 2.0, 'coo': 2.0,
                                'vp': 2.5, 'director': 2.0, 'manager': 1.5},
            'job_titles': {'engineer': 1.2, 'analyst': 1.0, 'specialist': 1.0},
            'company_terms': {'inc': 1.0, 'corp': 1.0, 'llc': 1.0, 'group': 0.8},
            'work_verbs': {'manage': 1.2, 'lead': 1.0, 'develop': 0.8, 
                          'optimize': 0.8, 'implement': 0.8}
        }
    }

    student_score = 0
    professional_score = 0
    
    # Scoring logic
    for term, weight in TERMS['student']['academic_roles'].items():
        if term in doc.text:
            student_score += weight
            
    for term, weight in {**TERMS['professional']['executive_titles'], 
                         **TERMS['professional']['job_titles']}.items():
        if term in doc.text:
            professional_score += weight
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            for term, weight in TERMS['student']['academic_orgs'].items():
                if term in ent.text.lower():
                    student_score += weight * (0.5 if professional_score > 2 else 1.0)
        
        if ent.label_ == 'DATE' and any(year in ent.text for year in ['2025', '2026', '2027']):
            student_score += 1.0
    
    for token in doc:
        if token.lemma_ in TERMS['student']['academic_verbs']:
            student_score += TERMS['student']['academic_verbs'][token.lemma_]
            
        if token.lemma_ in TERMS['professional']['work_verbs']:
            professional_score += TERMS['professional']['work_verbs'][token.lemma_]
    
    if professional_score >= 2.5:
        student_score *= 0.6
    
    return student_score > professional_score

def classify_students(input_json_path, output_json_path=None):
    """
    Classify students and add 'is_student' field to each profile
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Optional path for output JSON file
    
    Returns:
        Modified data with 'is_student' field added
    """
    with open(input_json_path) as f:
        try:
            # First try to load as list of dicts
            data = json.load(f)
            if isinstance(data, dict) and 'data' in data:
                # Handle case where data is wrapped in metadata
                profiles = data['data']
            elif isinstance(data, list):
                profiles = data
            else:
                raise ValueError("Unexpected JSON format")
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON file: {input_json_path}")
    
    # Process each profile and add classification
    for profile in profiles:
        if not isinstance(profile, dict):
            continue  # Skip non-dict items
            
        position_text = str(profile.get('position', ''))
        profile['is_student'] = analyze_student_status(position_text)
    
    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(profiles, f, indent=2)
        logging.info(f"Saved classified data to {output_json_path}")
    
    return profiles

def main():
    # Example usage:
    # classify_students('nyc_data.json', 'classified_data.json')
    pass

if __name__ == '__main__':
    main()

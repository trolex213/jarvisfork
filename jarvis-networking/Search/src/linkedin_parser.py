# Configuration

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from pydantic import BaseModel, field_validator, ConfigDict, Field
from datetime import datetime
import ollama
import re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed  # Add as_completed to imports
from tqdm import tqdm  # Add to imports

# Configure logging at module level
logger = logging.getLogger(__name__)

OLLAMA_MODEL = "gemma3:12b"
MAX_RETRIES = 3

# Token estimation constants (optimized for English resumes)
TOKEN_RATIOS = {
    'english': 3.8,  # chars per token (empirically measured for resumes)
    'technical': 2.5,  # for skills/education sections
    'sparse': 5.0  # for narrative text
}

def estimate_tokens(text: str) -> int:
    """Fast token estimation without full tokenization"""
    if not text:
        return 0
        
    # Count technical segments (bullets, ALL_CAPS, abbreviations)
    tech_segments = sum(
        1 for _ in re.finditer(r'\b[A-Z]{2,}\b|•|\d+[+\-*/]|\w\.\w', text)
    )
    tech_ratio = TOKEN_RATIOS['technical'] if tech_segments > 5 else TOKEN_RATIOS['english']
    
    # Estimate
    return int(len(text) / (
        tech_ratio * 0.3 +  # Weight technical sections
        TOKEN_RATIOS['sparse'] * 0.7  # Weight narrative
    ))

def read_mixed_json_file(json_file: Path) -> List:
    """Read and validate mixed JSON files from directory."""
    profiles = []
    error_count = 0
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle both wrapped and raw JSON formats
            if isinstance(data, dict) and 'data' in data:
                profile_data = data['data']
            else:
                profile_data = data
            
            if isinstance(profile_data, list):
                profiles.extend(profile_data)
            else:
                profiles.append(profile_data)
    except Exception as e:
        logger.error(f"Error reading JSON file: {str(e)}")
    
    return profiles
        
    
    
def generate_connection_points(resume_data):
    """Generate interesting connection points"""
    try:
        if not resume_data:
            logger.warning("Empty resume data provided")
            return None
            
        prompt = """give me 5 interesting things in this person's resume that could have a unique connection with other people. 
list in order of likelihood to connect (and how strong the connection will be). 
For the first two, always put university and high school.

Resume Data:
{resume_data}

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "connection_points": [
    {{
      "item": "description of item",
      "reason": "why it's interesting",
      "connection_strength": 0-10,
      "connection_potential": "description"
    }}
  ]
}}""".format(resume_data=json.dumps(resume_data, indent=2))
        
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                'num_gpu': -1,
                'num_ctx': 4096,
                'temperature': 0.5,
                'seed': 42
            }
        )
        
        raw_response = response['response'].strip()
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:-3].strip()
        
        return json.loads(raw_response)
        
    except Exception as e:
        logger.error(f"Connection generation failed: {str(e)}")
        if 'response' in locals():
            logger.debug(f"Raw response: {response['response']}")
        return None

async def batch_generate_embeddings(texts: List[str], model: str = "bge-m3", batch_size: int = 5) -> List[List[float]]:
    """Generate embeddings in batches"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                ollama.embeddings,
                model=model,
                prompt=text[:8192],
                options={'num_ctx': 4096}
            )
            for text in texts
        ]
        return [future.result()['embedding'] for future in futures]

def generate_embeddings(text: str, model: str = "bge-m3") -> List[float]:
    """Generate embeddings with better error handling"""
    try:
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
            
        response = ollama.embeddings(
            model=model,
            prompt=str(text)[:8192],  # Truncate to model max length
            options={'num_ctx': 4096}
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return []

class LinkedInProfile(BaseModel):
    """Flexible profile model that handles real-world data"""
    model_config = ConfigDict(extra="ignore")
    
    # All fields optional with defaults
    name: Optional[str] = None
    about: Optional[str] = None
    city: Optional[str] = None
    country_code: Optional[str] = None
    current_company_name: Optional[str] = None
    position: Optional[str] = None
    professional_keywords: Optional[str] = None
    personal_keywords: Optional[str] = None
    is_student: Optional[bool] = None
    generated_summary: Optional[str] = None
    experience_years: Optional[float] = None
    education: Optional[List[Dict]] = None
    experience: Optional[List[Dict]] = None
    organizations: Optional[List[Dict]] = None

    @field_validator('*', mode='before')
    @classmethod
    def handle_nulls(cls, value, info):
        if value is None:
            if info.field_name in ['education', 'experience', 'organizations']:
                return []
            if info.field_name == 'is_student':
                return False
            if info.field_name in ['experience_years', 'generated_summary']:
                return 0.0 if info.field_name == 'experience_years' else ""
        return value

def process_batch(batch: List[Dict]) -> List[Dict]:
    """Process a single batch of profiles"""
    batch_results = []
    for profile in batch:
        try:
            connections = generate_connection_points(profile)
            if connections:
                profile['connections'] = connections
                batch_results.append(profile)
        except Exception as e:
            logger.warning(f"Skipping profile due to error: {str(e)}")
    return batch_results

def process_profiles(data_path: Path, batch_size: int = 10) -> List[Dict]:
    """Process profiles with parallel batch execution"""
    try:
        data = read_mixed_json_file(data_path)
        if not data:
            logger.error("No profiles loaded")
            return []

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Submit all batches for processing
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                futures.append(executor.submit(process_batch, batch))
            
            # Process completed batches
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    logger.info(f"Processed batch of {len(batch_results)} profiles")
                except Exception as e:
                    logger.error(f"Batch failed: {str(e)}")
        
        return results
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process LinkedIn profiles and generate connection analysis')
    parser.add_argument('--input', required=True, help='Path to input JSON file with profiles')
    parser.add_argument('--output', required=True, help='Path to save processed output JSON')
    parser.add_argument('--batch_size', type=int, default=10, 
                      help='Number of profiles to process simultaneously (default: 10)')
    args = parser.parse_args()
    
    try:
        # Process profiles with specified batch size
        results = process_profiles(Path(args.input), batch_size=args.batch_size)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Successfully processed {len(results)} profiles. Saved to {args.output}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)
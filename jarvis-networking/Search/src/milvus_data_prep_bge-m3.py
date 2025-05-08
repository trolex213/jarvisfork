from collections import defaultdict
from typing import Dict, Any, Optional, Union, List
import json
from tqdm import tqdm
import ollama
import torch
from data_standardizer import DataStandardizer
import argparse
import os
import sys
import logging
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymilvus import CollectionSchema, FieldSchema, DataType

# BGE-M3 configuration with Ollama
OLLAMA_MODEL = "bge-m3"  # Assumes you've pulled this model via `ollama pull bge-m3`
NETWORK_INSTRUCTION = "Represent this LinkedIn profile for finding relevant professional connections: "

# Ollama client setup
client = ollama.Client()

def safe_get(data: Union[Dict, List], *keys, default=None):
    """
    Safely get nested value from dict/list with fallback.
    """
    try:
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
            elif isinstance(data, list) and isinstance(key, int):
                data = data[key] if key < len(data) else None
            else:
                return default
            if data is None:
                return default
        return data if data is not None else default
    except (KeyError, IndexError, AttributeError, TypeError):
        return default

class NetworkProfileProcessor:
    """Processes LinkedIn profiles using BGE-M3 via Ollama."""
    
    def __init__(self, schema_config: Optional[str] = None):
        self.connection_weights = {
            'education': 0.4,
            'organization': 0.35,
            'shared_connections': 0.25
        }
        self.schema = self._load_schema(schema_config) if schema_config else None
        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using BGE-M3 via Ollama."""
        if not text.strip():
            self.logger.warning("Empty embedding text")
            return []
            
        try:
            response = client.embeddings(
                model=OLLAMA_MODEL,
                prompt=NETWORK_INSTRUCTION + text,
                options={
                    'num_ctx': 8192,  # BGE-M3 supports longer context
                    'temperature': 0.1
                }
            )
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return []

    # [Rest of the class methods remain the same as original]
    def _load_schema(self, config_path: str):
        """Load Milvus schema configuration."""
        with open(config_path) as f:
            return json.load(f)

    def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text."""
        if not text or not isinstance(text, str):
            return ""
        try:
            tokens = word_tokenize(text.lower())
            keywords = [
                word for word in tokens 
                if word.isalpha() and word not in stopwords.words('english')
            ]
            return ", ".join(sorted(set(keywords))[:5])
        except Exception:
            return ""

    def prepare_milvus_record(self, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare profile data for Milvus."""
        if not isinstance(profile, dict):
            self.logger.error("Invalid profile format")
            return None
            
        profile_id = profile.get("id") or str(hash(json.dumps(profile, sort_keys=True)))
        
        return {
            "id": int(profile_id) if str(profile_id).isdigit() else abs(hash(str(profile_id))),
            "vector": self.generate_embedding(profile.get("embedding_text", "")),
            "name": str(profile.get("name", "")).strip(),
            "city": str(profile.get("city", "")).strip(),
            "country_code": str(profile.get("country_code", "")).strip()[:2],
            "current_company_name": str(profile.get("current_company", {}).get("name", "")).strip(),
            "position": str(profile.get("position", "")).strip(),
            "keywords": self._extract_keywords(profile.get("about", "")),
            "is_student": bool(profile.get("is_student", False)),
            "education_count": len(profile.get("education", [])),
            "experience_years": self._calculate_experience_years(profile.get("experience", [])),
            "metadata": {"raw": profile}
        }

    def _calculate_experience_years(self, experiences: List[Dict]) -> float:
        """Calculate total professional experience in years."""
        total = 0.0
        for exp in experiences:
            if exp.get("duration"):
                try:
                    total += float(exp["duration"].split(" ")[0])
                except (ValueError, IndexError, AttributeError):
                    continue
        return round(total, 1)

# [Rest of the file remains the same as original]

if __name__ == "__main__":
    """Command line interface for Milvus data preparation."""
    parser = argparse.ArgumentParser(description='Prepare LinkedIn profile data for Milvus using BGE-M3')
    parser.add_argument('--input', required=True, help='Input JSON file with profiles')
    parser.add_argument('--output', required=True, help='Output JSON file for Milvus')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--schema', help='Path to Milvus schema config')
    
    args = parser.parse_args()
    
    processor = NetworkProfileProcessor(args.schema)
    
    with open(args.input) as f:
        profiles = json.load(f)
    
    results = []
    for i in tqdm(range(0, len(profiles), args.batch_size), desc="Processing profiles"):
        batch = profiles[i:i+args.batch_size]
        for profile in batch:
            processed = processor.prepare_milvus_record(profile)
            if processed:
                results.append(processed)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Successfully processed {len(results)} profiles. Saved to {args.output}")

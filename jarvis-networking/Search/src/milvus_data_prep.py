from collections import defaultdict
from typing import Dict, Any, Optional, Union, List
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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

# Initialize BAAI model with ROCm optimizations
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

MODEL = SentenceTransformer(
    'BAAI/bge-large-en-v1.5',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
NETWORK_INSTRUCTION = "Represent this LinkedIn profile for finding relevant professional connections: "

def safe_get(data: Union[Dict, List], *keys, default=None):
    """
    Safely get nested value from dict/list with fallback.
    
    Args:
        data: Input dictionary or list
        *keys: Nested keys/indices to traverse
        default: Fallback value if path not found
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
    """
    Processes LinkedIn profiles for optimized network matching in Milvus.
    
    Handles:
    - Embedding generation
    - Data validation
    - Schema creation
    - Experience calculation
    """
    
    def __init__(self, schema_config: Optional[str] = None):
        """
        Initialize processor with optional schema configuration.
        
        Args:
            schema_config: Path to Milvus schema configuration file
        """
        self.connection_weights = {
            'education': 0.4,
            'organization': 0.35,
            'shared_connections': 0.25
        }
        self.schema = self._load_schema(schema_config) if schema_config else None
        self.logger = logging.getLogger(__name__)

    def _load_schema(self, config_path: str):
        """
        Load Milvus schema configuration from file.
        
        Args:
            config_path: Path to schema configuration file
            
        Returns:
            Loaded schema configuration
        """
        with open(config_path) as f:
            return json.load(f)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding with instruction tuning.
        
        Args:
            text: Input text for embedding generation
            
        Returns:
            Generated embedding vector
        """
        if not text.strip():
            self.logger.warning("Empty embedding text")
            return []
        return MODEL.encode(
            [NETWORK_INSTRUCTION + text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).cpu().numpy()[0].tolist()

    def _extract_keywords(self, text: str) -> str:
        """
        Extract keywords using basic NLP without external dependencies.
        
        Args:
            text: Input text for keyword extraction
            
        Returns:
            Comma-separated keywords
        """
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Simple word tokenization and filtering
            tokens = word_tokenize(text.lower())
            keywords = [
                word for word in tokens 
                if word.isalpha() and word not in stopwords.words('english')
            ]
            return ", ".join(sorted(set(keywords))[:5])  # Return top 5 unique keywords
        except Exception:
            return ""

    def prepare_milvus_record(self, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepare profile data with basic null checks.
        
        Args:
            profile: Input profile data
            
        Returns:
            Prepared profile data or None if invalid
        """
        if not isinstance(profile, dict):
            self.logger.error("Invalid profile format")
            return None
            
        # Make ID check more resilient - generate a hash if missing
        profile_id = profile.get("id")
        if not profile_id:
            profile_id = str(hash(json.dumps(profile, sort_keys=True)))
            self.logger.warning(f"Generated profile ID: {profile_id}")
            
        if not isinstance(profile.get("name"), str):
            self.logger.warning("Invalid name field")
            return None
            
        if not isinstance(profile.get("position"), str):
            self.logger.warning("Invalid position field")
            return None
            
        embedding_text = profile.get("embedding_text", "")
        if not embedding_text or not isinstance(embedding_text, str):
            self.logger.warning("Invalid embedding text")
            return None
            
        # Ensure education and experience are lists
        education = profile.get("education")
        if education is None:
            education = []
            
        experience = profile.get("experience")
        if experience is None:
            experience = []
            
        return {
            "id": int(profile_id) if str(profile_id).isdigit() else abs(hash(str(profile_id))),
            "vector": self.generate_embedding(embedding_text),
            "name": str(profile["name"]).strip(),
            "city": str(profile.get("city", "")).strip(),
            "country_code": str(profile.get("country_code", "")).strip()[:2],
            "current_company_name": str(profile.get("current_company", {}).get("name", "")).strip(),
            "position": str(profile["position"]).strip(),
            "keywords": self._extract_keywords(profile.get("about", "")),
            "is_student": bool(profile.get("is_student", False)),
            "education_count": len(education),
            "experience_years": self._calculate_experience_years(experience),
            "metadata": {"raw": profile}
        }
            
    def _calculate_experience_years(self, experiences: List[Dict]) -> float:
        """
        Calculate total professional experience in years.
        
        Args:
            experiences: List of experience dicts
            
        Returns:
            Total years of experience (rounded to 1 decimal)
        """
        total = 0.0
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
                
            duration = exp.get("duration")
            if not duration:
                continue
                
            try:
                # Handle formats like "1996 - Present 29" or "Jun 2021 Jun 2022 1"
                if " - " in duration or len(duration.split()) >= 3:
                    parts = duration.split()
                    if len(parts) >= 3 and parts[-1].isdigit():
                        total += float(parts[-1])
                # Handle simple "X years" format
                elif "year" in duration:
                    total += float(duration.split("year")[0].strip())
            except (ValueError, AttributeError):
                self.logger.warning(f"Could not parse duration: {duration}")
                continue
                
        return round(total, 1)
        
    def get_hybrid_schema(self, dim: int = 768) -> CollectionSchema:
        """
        Generate optimized Milvus schema for hybrid search.
        
        Args:
            dim: Embedding dimension size
            
        Returns:
            Configured CollectionSchema object
        """
        return CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="country_code", dtype=DataType.VARCHAR, max_length=2),
                FieldSchema(name="current_company_name", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="is_student", dtype=DataType.BOOL),
                FieldSchema(name="education_count", dtype=DataType.INT16),
                FieldSchema(name="experience_years", dtype=DataType.FLOAT),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ],
            description="Hybrid search schema for professional profiles"
        )

    def _extract_network_graph(self, profile: Dict) -> Dict:
        """
        Extract network graph connections with error handling.
        
        Args:
            profile: Input profile data
            
        Returns:
            Extracted network graph data
        """
        try:
            education = profile.get("education", [])
            if not isinstance(education, list):
                education = []
                
            experience = profile.get("experience", [])
            if not isinstance(experience, list):
                experience = []
                
            return {
                "nodes": list({
                    *["profile"],
                    *[f"edu_{edu.get('school', 'unknown')}" for edu in education if isinstance(edu, dict)],
                    *[f"org_{exp.get('company', 'unknown')}" for exp in experience if isinstance(exp, dict)]
                }),
                "connections": [
                    {"source": "profile", "target": f"edu_{edu.get('school', 'unknown')}", 
                     "weight": self.connection_weights['education']}
                    for edu in education if isinstance(edu, dict)
                ] + [
                    {"source": "profile", "target": f"org_{exp.get('company', 'unknown')}", 
                     "weight": self.connection_weights['organization']}
                    for exp in experience if isinstance(exp, dict)
                ]
            }
        except Exception as e:
            self.logger.warning(f"Network graph extraction failed: {str(e)}")
            return {}
            
    def _extract_temporal_features(self, profile: Dict) -> Dict:
        """
        Extract temporal features with error handling.
        
        Args:
            profile: Input profile data
            
        Returns:
            Extracted temporal features
        """
        try:
            education = profile.get("education", [])
            if not isinstance(education, list):
                education = []
                
            experience = profile.get("experience", [])
            if not isinstance(experience, list):
                experience = []
                
            return {
                "education": [
                    {
                        "school": edu.get("school", "unknown"),
                        "start_year": edu.get("start_year"),
                        "end_year": edu.get("end_year")
                    }
                    for edu in education if isinstance(edu, dict)
                ],
                "experience": [
                    {
                        "company": exp.get("company", "unknown"),
                        "start_year": exp.get("start_year"),
                        "end_year": exp.get("end_year")
                    }
                    for exp in experience if isinstance(exp, dict)
                ]
            }
        except Exception as e:
            self.logger.warning(f"Temporal feature extraction failed: {str(e)}")
            return {}

    def _normalize_degree_name(self, degree: Optional[str]) -> Optional[str]:
        """
        Normalize degree name for consistency.
        
        Args:
            degree: Input degree name
            
        Returns:
            Normalized degree name
        """
        if not degree:
            return None
        degree = degree.replace("School of", "").replace("Business", "").strip()
        return re.sub(r"\s+", " ", degree)
    
    def _calculate_total_education_years(self, profile: Dict) -> float:
        """
        Calculate total education years.
        
        Args:
            profile: Input profile data
            
        Returns:
            Total education years
        """
        # First try structured data
        structured_years = sum((safe_get(edu, "duration_years", 0) or 0) 
                              for edu in safe_get(profile, "education", []))
        if structured_years > 0:
            return structured_years
            
        # Fall back to text extraction
        summary = safe_get(profile, "generated_summary", "")
        if "Duration:" in summary:
            matches = re.findall(r"Duration: (\d+) years (\d+) months", summary)
            return sum(float(y) + float(m)/12 for y, m in matches)
        return 0

    def _calculate_boost_weights(self, profile: Dict) -> Dict:
        """
        Calculate field-specific boost weights for hybrid search.
        
        Args:
            profile: Input profile data
            
        Returns:
            Boost weights for each field
        """
        return {
            "education": min(len(profile.get("education", [])), 5) * 0.2,
            "experience": sum(
                self._parse_duration_years(exp["duration"]) 
                for exp in profile.get("experience", [])
            ) * 0.1,
            "skills": len(profile.get("skills", [])) * 0.05
        }

    def _parse_duration_years(self, duration: str) -> int:
        """
        Parse experience duration into years.
        
        Args:
            duration: Input duration string
            
        Returns:
            Parsed duration in years
        """
        if "year" in duration:
            return int(duration.split("year")[0].strip())
        return 0

    def _normalize_company(self, company: Dict) -> Dict:
        """
        Standardize company structure.
        
        Args:
            company: Input company data
            
        Returns:
            Standardized company data
        """
        if not isinstance(company, dict):
            return {"name": "", "industry": ""}
        return {
            "name": safe_get(company, "name", "").strip(),
            "industry": safe_get(company, "industry", "").strip()
        }

    def process_dataset(self, input_path: str, output_path: str, batch_size: int = 500, schema_path: str = None):
        """
        Process profile dataset for Milvus import.
        
        Args:
            input_path: Path to input JSON profiles
            output_path: Path to save processed output
            batch_size: Number of profiles per processing batch
            schema_path: Optional path to Milvus schema config
        """
        if schema_path:
            self._load_schema(schema_path)
            
        total_processed = 0
        total_skipped = 0
        
        try:
            with open(input_path) as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                self.logger.error(f"Input data must be a list of profiles, got {type(data)}")
                return
                
            self.logger.info(f"Starting processing of {len(data)} profiles")
            
            with open(output_path, 'w') as outfile:
                batch = []
                
                for i, profile in enumerate(tqdm(data, desc="Processing profiles")):
                    processed = self.prepare_milvus_record(profile)
                    if processed:
                        batch.append(processed)
                        total_processed += 1
                    else:
                        total_skipped += 1
                        
                    if len(batch) >= batch_size:
                        json.dump(batch, outfile)
                        outfile.write('\n')
                        batch = []
                
                if batch:
                    json.dump(batch, outfile)
                    
            self.logger.info(f"Completed processing. Success: {total_processed}, Skipped: {total_skipped}")
            
        except Exception as e:
            self.logger.error(f"Failed to process dataset: {str(e)}")
            raise

def process_dataset(input_path: str, output_path: str, batch_size: int = 500, schema_path: str = None):
    """
    Process profile dataset for Milvus import.
    
    Args:
        input_path: Path to input JSON profiles
        output_path: Path to save processed output
        batch_size: Number of profiles per processing batch
        schema_path: Optional path to Milvus schema config
    """
    processor = NetworkProfileProcessor(schema_path)
    processor.process_dataset(input_path, output_path, batch_size, schema_path)

if __name__ == "__main__":
    """
    Command line interface for Milvus data preparation.
    Usage: python milvus_data_prep.py [OPTIONS]
    
    Options:
      --input: Input JSON file path
      --output: Output JSON file path
      --batch_size: Processing batch size
      --schema: Milvus schema config path
    """
    logging.basicConfig(level=logging.ERROR)  # Only show ERROR level and above
    parser = argparse.ArgumentParser(description='Prepare LinkedIn data for Milvus')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--schema', help='Path to Milvus schema config')
    args = parser.parse_args()
    
    # Initialize processor with suppressed warnings
    processor = NetworkProfileProcessor(args.schema)
    processor.logger.setLevel(logging.ERROR)
    
    process_dataset(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        schema_path=args.schema
    )
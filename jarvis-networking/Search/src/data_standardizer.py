# data_standardizer.py
import argparse
import json
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import logging

DATE_FORMATS = [
    ('%Y-%m', 'partial_month'),
    ('%m/%Y', 'slash_date'),
    ('%Y', 'year_only'),
    ('%b %Y', 'short_month'),
    ('%B %Y', 'long_month')
]

class DataCleaner:
    """Handles data sanitization and type validation"""
    
    @staticmethod
    def clean_nan(data: Any) -> Any:
        """Recursively replace NaN/None with None and clean data structures"""
        if isinstance(data, dict):
            return {k: DataCleaner.clean_nan(v) for k, v in data.items()}
        if isinstance(data, list):
            return [DataCleaner.clean_nan(item) for item in data]
        if isinstance(data, float) and data != data:  # NaN check
            return None
        return data

    @staticmethod
    def validate_structure(data: Any) -> Optional[Union[dict, List[dict]]]:
        """Ensure data is a dict or list of dicts, returns cleaned structure or None"""
        cleaned = DataCleaner.clean_nan(data)
        
        if isinstance(cleaned, dict):
            return cleaned
        if isinstance(cleaned, list):
            return [item for item in cleaned if isinstance(item, dict)]
        return None

class DateNormalizer:
    """Handles date parsing and normalization with pattern tracking"""
    
    def __init__(self):
        self.date_patterns = defaultdict(lambda: defaultdict(int))
        
    def normalize_date(self, field_path: str, date_str: str) -> str:
        """Normalize date string to ISO format or return original if parsing fails"""
        try:
            dt = datetime.fromisoformat(date_str)
            return dt.date().isoformat()
        except ValueError:
            for fmt, pattern_name in DATE_FORMATS:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    iso_date = dt.date().isoformat()
                    self.date_patterns[field_path][pattern_name] += 1
                    return iso_date
                except ValueError:
                    continue
            self.date_patterns[field_path]['unrecognized'] += 1
            return date_str

class DataStandardizer:
    """Main class for normalizing and validating profile data"""
    
    def __init__(self, schema_path: Optional[str] = None, verbose: bool = False):
        self.schema = None
        self.date_normalizer = DateNormalizer()
        self.validation_errors = []
        self.verbose = verbose
        
        if schema_path:
            self.load_schema(schema_path)

    def load_schema(self, schema_path: str):
        """Load JSON schema for validation"""
        with open(schema_path) as f:
            self.schema = json.load(f)

    def normalize_profile(self, profile: Dict) -> Dict:
        """Normalize profile data types while preserving structure"""
        if not isinstance(profile, dict):
            if self.verbose:
                print(f"Debug: Invalid profile type - {type(profile)}")
            raise ValueError("Profile must be a dictionary")

        normalized = {}
        for key, value in profile.items():
            try:
                if value is None:
                    normalized[key] = None
                elif isinstance(value, str):
                    normalized[key] = str(value).strip()
                elif isinstance(value, (int, float)):
                    normalized[key] = float(value)
                elif isinstance(value, bool):
                    normalized[key] = bool(value)
                elif isinstance(value, list):
                    normalized[key] = [self.normalize_profile(x) if isinstance(x, dict) else x for x in value]
                elif isinstance(value, dict):
                    normalized[key] = self.normalize_profile(value)
                else:
                    normalized[key] = str(value)  # Fallback to string
                    
                if self.verbose:
                    print(f"Debug: Normalized {key} ({type(value)} -> {type(normalized[key])})")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Debug: Failed to normalize {key}: {str(e)}")
                normalized[key] = None  # Fallback to None

        return normalized

    def normalize_dates(self, profile: Dict) -> Dict:
        """Public alias for normalize_profile"""
        return self.normalize_profile(profile)

    def _normalize_dates(self, profile: Dict) -> Dict:
        """Normalize dates in a profile with detailed logging"""
        if not isinstance(profile, dict):
            return {}
            
        return {
            key: self._process_value(key, value)
            for key, value in profile.items()
        }

    def _process_value(self, key: str, value: Any) -> Any:
        """Recursive value processor with error handling"""
        try:
            if isinstance(value, dict):
                return self._normalize_dates(value)
            if isinstance(value, list):
                return [self._process_value(key, item) for item in value]
            if isinstance(value, str) and 'date' in key.lower():
                return self.date_normalizer.normalize_date(key, value)
            return value
        except Exception as e:
            return value

    def standardize_for_resume(self, profile: Dict) -> Dict:
        """Specialized standardization for resume matching"""
        standardized = {
            "name": self._standardize_name(profile.get("name")),
            "experience": [self._standardize_experience(exp) for exp in profile.get("experience", [])],
            "education": [self._standardize_education(edu) for edu in profile.get("education", [])],
            "skills": self._standardize_skills(profile.get("skills", [])),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add resume-specific fields
        if "resume_parser_version" not in profile:
            standardized["resume_parser_version"] = "1.0"
            
        return standardized

    def _standardize_experience(self, exp: Dict) -> Dict:
        """Standardize experience to match resume format"""
        return {
            "company": exp.get("company", "").strip().title(),
            "title": exp.get("title", "").strip().title(),
            "duration_years": self._parse_duration(exp.get("duration", "")),
            "skills": self._standardize_skills(exp.get("skills", []))
        }

    def _standardize_education(self, edu: Dict) -> Dict:
        """Standardize education to match resume format"""
        return {
            "institution": edu.get("school", "").strip().title(),
            "degree": edu.get("degree", "").strip().upper(),
            "year": self._parse_education_year(edu.get("end_year"))
        }

    def _standardize_name(self, name: str) -> str:
        """Standardize name to match resume format"""
        return name.strip().title()

    def _standardize_skills(self, skills: List[str]) -> List[str]:
        """Standardize skills to match resume format"""
        return [skill.strip().lower() for skill in skills]

    def _parse_duration(self, duration: str) -> int:
        """Parse duration string to years"""
        # Implement duration parsing logic here
        return 0

    def _parse_education_year(self, year: str) -> int:
        """Parse education year string to integer"""
        # Implement education year parsing logic here
        return 0

class DataProcessor:
    """Handles batch processing of input data"""
    
    def __init__(self, standardizer: DataStandardizer, output_path: str, batch_size: int = 1000):
        self.standardizer = standardizer
        # output_path is always treated as file path
        self.output_path = output_path
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        self.batch_size = batch_size
        self.processed = 0
        self.logger = logging.getLogger(__name__)

    def process(self, input_path: str, output_path: str = None):
        """Main processing method with full error handling"""
        # Use provided output_path or fall back to self.output_path
        output_file = output_path if output_path else self.output_path
        with open(input_path) as fin, open(output_file, 'w') as fout:
            batch = []
            for line in self._read_lines(fin):
                cleaned_data = self._process_line(line)
                if cleaned_data:
                    batch.extend(cleaned_data)
                    if len(batch) >= self.batch_size:
                        self._write_batch(fout, batch)
                        batch = []
            
            if batch:
                self._write_batch(fout, batch)

    def _read_lines(self, file_handle):
        """Read lines with progress tracking"""
        return tqdm(file_handle, desc="Processing records", unit="rec")

    def _process_line(self, line: str) -> Optional[dict]:
        """Process a single line of JSON with robust error handling."""
        try:
            profile = json.loads(line)
            return self.standardizer.normalize_profile(profile)
        except json.JSONDecodeError as e:
            try:
                # Try fixing common issues
                fixed = line.strip()
                fixed = re.sub(r'(\{|\,)\s*(\w+)\s*:', lambda m: f'{m.group(1)}"{m.group(2)}":', fixed)
                fixed = re.sub(r',\s*([}\]])\s*$', r'\1', fixed)
                
                profile = json.loads(fixed)
                return self.standardizer.normalize_profile(profile)
                
            except Exception:
                # Error logs go in same directory as output file
                error_log = os.path.join(os.path.dirname(self.output_path), "malformed_lines.log")
                with open(error_log, "a") as f:
                    f.write(f"=== MALFORMED ITEM ===\n{line}\n")
                    f.write(f"=== ERROR DETAILS ===\n{str(e)}\n\n")
                return None

    def _write_batch(self, fout, batch: List[Dict]):
        if not batch:
            return

        # Debug: Print type and structure of first few records
        for idx, record in enumerate(batch[:5]):
            if self.standardizer.verbose:
                print(f"[DEBUG] Record {idx} type: {type(record)}, content: {repr(record)[:300]}")

        valid_records = []
        for idx, record in enumerate(batch):
            if not isinstance(record, dict):
                if self.standardizer.verbose:
                    print(f"⚠️ Invalid record type at index {idx}: {type(record).__name__}")
                continue
            if not isinstance(record.get('metadata'), dict):
                if self.standardizer.verbose:
                    print(f"⚠️ Missing metadata dict in record {idx}")
                continue
            valid_records.append(record)

        try:
            json.dump(valid_records, fout)
            fout.write('\n')
            self.processed += len(valid_records)
            if self.standardizer.verbose:
                print(f"Successfully wrote {len(valid_records)} records")
        except Exception as e:
            if valid_records:
                if self.standardizer.verbose:
                    print(f"First valid record sample: {valid_records[0]}")

def generate_keyfield_report(input_path: str, output_dir: str, structure_report_path: Optional[str] = None) -> str:
    """Generate report using structure report insights"""
    # Load structure report if available
    structure_data = {}
    if structure_report_path:
        report_file = os.path.join(structure_report_path, "raw_structure_report.json")
        try:
            with open(report_file) as f:
                structure_data = json.load(f)
        except Exception as e:
            warnings.warn(f"Couldn't load structure report: {str(e)}")
    
    # Get key fields from structure report or use defaults
    KEY_FIELDS = structure_data.get('top_fields', [
        'name', 'company', 'title',
        'skills', 'education.degree',
        'experience.company', 'experience.title'
    ])
    
    # Create report directory
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "keyfield_analysis.html")
    
    # Generate HTML report
    with open(report_path, 'w') as f:
        f.write(f"""
        <h1>Key Field Analysis Report</h1>
        <p>Generated from structure report: {report_file if structure_report_path else 'N/A'}</p>
        """)
        
        for field in KEY_FIELDS:
            field_stats = structure_data.get('field_stats', {}).get(field, {})
            f.write(f"""
            <div style='margin:20px; padding:15px; border:1px solid #ddd;'>
                <h2>{field}</h2>
                <p><b>Presence:</b> {field_stats.get('presence_pct', 0):.1f}% of records</p>
                <p><b>Type Distribution:</b> {field_stats.get('type_distribution', {})}</p>
                <p><b>Top Values:</b> {field_stats.get('top_values', [])[:5]}</p>
            </div>
            """)
    
    return report_path

def _get_nested(data: dict, path: str) -> Any:
    """Get nested value from dict"""
    keys = path.split('.')
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data

def _detect_value_type(values: list) -> str:
    """Categorize field values"""
    samples = values[:100]
    
    # Date detection
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%b %Y"]
    date_count = 0
    for v in samples:
        if isinstance(v, (int, float)):
            date_count += 1
        else:
            for fmt in date_formats:
                try:
                    datetime.strptime(str(v), fmt)
                    date_count += 1
                    break
                except ValueError:
                    pass
    if date_count / len(samples) > 0.7:
        return "date"
    
    # Numeric detection
    if all(isinstance(v, (int, float)) for v in samples):
        return "numeric"
    
    # Categorical detection
    unique_ratio = len(set(samples)) / len(samples)
    if unique_ratio < 0.3:
        return "categorical"
    
    return "text"

def _get_standardization_recommendation(result: dict) -> str:
    """Generate standardization recommendation"""
    if result['likely_type'] == 'date':
        return "Use date normalization"
    elif result['likely_type'] == 'numeric':
        return "Use numeric formatting"
    elif result['likely_type'] == 'categorical':
        return "Use categorical encoding"
    else:
        return "Use text normalization"
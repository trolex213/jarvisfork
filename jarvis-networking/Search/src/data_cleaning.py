# data_cleaning.py
import pandas as pd
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from classify_student import classify_students
from clean_urls import clean_profile_urls
from profile_summarizer import process_dataset as summarize_profiles
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import psutil
from typing import Dict, Optional, Any
from data_standardizer import DataStandardizer, DataProcessor

class MemoryTracker:
    """
    Tracks system memory usage.
    """
    def get_usage(self):
        """
        Get current CPU and GPU usage.
        
        Returns:
            Dict with 'cpu' and 'gpu' usage percentages
        """
        cpu_usage = psutil.cpu_percent()
        gpu_usage = 0  # Replace with actual GPU usage tracking
        return {'cpu': cpu_usage, 'gpu': gpu_usage}

class PipelineProgress:
    """
    Enhanced progress tracker with memory monitoring.
    """
    def __init__(self, total_steps):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of pipeline steps
        """
        self.current = 0
        self.total = total_steps
        self.start_time = time.time()
        self.mem_tracker = MemoryTracker()
        
    def update(self, stage: str):
        """
        Update progress tracker.
        
        Args:
            stage: Current pipeline stage
        """
        self.current += 1
        mem = self.mem_tracker.get_usage()
        elapsed = time.time() - self.start_time
        
        print(f"""
[Stage {self.current}/{self.total}] {stage}
Time: {elapsed:.1f}s | CPU: {mem['cpu']:.1f}% | GPU: {mem['gpu']:.1f}%
""")

def filter_nyc_region(input_json_path, output_json_path=None):
    """
    Filter profiles to only include those from NYC metro area.
    
    Args:
        input_json_path: Path to input JSON file with raw profiles
        output_json_path: Optional path to save filtered profiles
        
    Returns:
        Dict containing:
        - metadata: Processing info and record count
        - data: List of filtered profiles
    """
    nyc_locations = ['new york', 'nyc', 'brooklyn', 'queens', 'manhattan',
                   'bronx', 'staten island', 'jersey city', 'hoboken',
                   'newark', 'westchester']
    
    # Fields to preserve in output
    KEEP_FIELDS = {
        'about', 'city', 'country_code', 'current_company',
        'current_company_name', 'education', 'educations_details', 'experience',
        'honors_and_awards', 'location', 'name', 'organizations', 'position',
        'projects', 'publications', 'similar_profiles', 'url',
        'volunteer_experience'
    }
    
    with open(input_json_path) as f:
        data = json.load(f)
    
    # Handle both wrapped and raw JSON formats
    profiles = data['data'] if isinstance(data, dict) and 'data' in data else data
    
    filtered_profiles = []
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
            
        # Check location fields for NYC match
        location_fields = [
            str(profile.get('location', '')).lower(),
            str(profile.get('city', '')).lower()
        ]
        
        if any(nyc_loc in loc_field for nyc_loc in nyc_locations for loc_field in location_fields):
            # Filter profile to only keep specified fields
            filtered_profile = {k: v for k, v in profile.items() if k in KEEP_FIELDS}
            filtered_profiles.append(filtered_profile)
    
    result = {
        'metadata': {
            'processing_date': datetime.now().isoformat(),
            'input_file': str(input_json_path),
            'nyc_locations': nyc_locations,
            'record_count': len(filtered_profiles)
        },
        'data': filtered_profiles
    }
    
    if output_json_path:
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        logging.info(f"Saved filtered data to {output_json_path}")
    
    return result

def process_pipeline(input_path: str, filtered_output_path: str, classified_output_path: str,
                   cleaned_output_path: str = None, summarized_output_path: str = None) -> Dict[str, Any]:
    """
    Execute the complete LinkedIn profile processing pipeline.
    
    The pipeline consists of:
    1. Filtering - Select profiles from NYC region
    2. Cleaning - Standardize URLs and profile formats
    3. Classification - Identify student profiles
    4. Summarization - Generate condensed profile representations
    5. Structure Analysis - Analyze JSON structure and generate Milvus schema
    
    Args:
        input_path: Path to raw LinkedIn profile data
        filtered_output_path: Output path for NYC-filtered profiles
        classified_output_path: Output path for student-classified profiles
        cleaned_output_path: Optional path for cleaned profiles (default: None)
        summarized_output_path: Optional path for summarized profiles (default: None)
        
    Returns:
        Dictionary containing:
        - status: Pipeline completion status
        - analysis: Path to structure analysis JSON (if summarization performed)
        - schema: Path to Milvus schema JSON (if summarization performed)
    """
    try:
        # Standard pipeline steps
        filter_result = filter_nyc_region(input_path, filtered_output_path)
        clean_result = clean_profile_urls(filtered_output_path, cleaned_output_path)
        
        # Standardization
        standardized_path = os.path.join(os.path.dirname(cleaned_output_path), "standardized_profiles.json")
        normalize_profiles(cleaned_output_path, standardized_path)
        
        # First classify profiles
        classify_result = classify_students(standardized_path, classified_output_path)
        
        # Then summarize classified profiles
        if summarized_output_path:
            summary_result = summarize_profiles(
                classified_output_path,
                summarized_output_path
            )
            
            if not summary_result:
                raise ValueError("Profile summarization failed - empty result")
                
            # Prepare Milvus data from classified profiles
            milvus_output_path = os.path.join(os.path.dirname(summarized_output_path), 
                                            "milvus_" + os.path.basename(summarized_output_path))
            schema_path = os.path.join(os.path.dirname(summarized_output_path), "milvus_schema.json")
            
            # Generate schema
            from pymilvus import CollectionSchema, FieldSchema, DataType
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="country_code", dtype=DataType.VARCHAR, max_length=2),
                    FieldSchema(name="current_company_name", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="is_student", dtype=DataType.BOOL),
                    FieldSchema(name="education_count", dtype=DataType.INT16),
                    FieldSchema(name="experience_years", dtype=DataType.FLOAT),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ],
                description="Hybrid search schema for professional profiles"
            )
            
            with open(schema_path, 'w') as f:
                json.dump({"fields": [f.to_dict() for f in schema.fields]}, f, indent=2)
            
            os.system(f"python milvus_data_prep.py --input {summarized_output_path} --output {milvus_output_path} --batch_size 32 --schema {schema_path}")
            
            return {
                'status': 'complete',
                'analysis': summary_result.get('analysis') if summary_result else None,
                'schema': schema_path,
                'milvus_data': milvus_output_path
            }
        
        return {"status": "completed"}
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

def clean_dataset(input_path: str, output_path: str):
    """
    Clean raw LinkedIn data by:
    1. Filtering NYC profiles
    2. Classifying students
    3. Cleaning URLs
    """
    # Step 1: Filter NYC profiles
    filtered_path = "temp_filtered.json"
    filter_nyc_region(input_path, filtered_path)
    
    # Step 2: Classify students
    classified_path = "temp_classified.json"
    classify_students(filtered_path, classified_path)
    
    # Step 3: Clean URLs
    clean_profile_urls(classified_path, output_path)
    
    # Clean up temp files
    import os
    os.remove(filtered_path)
    os.remove(classified_path)

def execute_parallel_steps(input_path: str, output_dir: str):
    """
    Execute pipeline steps in parallel with progress tracking.
    
    Args:
        input_path: Path to raw input data
        output_dir: Directory for all output files
    """
    # Check if input is already standardized
    if 'standardized' in input_path.lower():
        standardized_path = input_path
    else:
        filtered_path = os.path.join(output_dir, "filtered_profiles.json")
        cleaned_path = os.path.join(output_dir, "cleaned_profiles.json") 
        standardized_path = os.path.join(output_dir, "standardized_profiles.json")
        
        # Only run initial steps if needed
        if not os.path.exists(filtered_path):
            filter_nyc_region(input_path, filtered_path)
        if not os.path.exists(cleaned_path):
            clean_profile_urls(filtered_path, cleaned_path)
        
        normalize_profiles(cleaned_path, standardized_path)
    
    classified_path = os.path.join(output_dir, "classified_profiles.json")
    
    # Only proceed if standardization succeeded
    if os.path.exists(standardized_path):
        classify_students(standardized_path, classified_path)
    
    return classified_path

def process_in_parallel(data, func, batch_size=5000, workers=None):
    """
    Process data in parallel batches with progress bar.
    
    Args:
        data: Input data to process
        func: Processing function to apply
        batch_size: Number of items per batch
        workers: Number of worker threads
    """
    import multiprocessing
    import sys
    from itertools import islice
    from tqdm.auto import tqdm
    
    # Configure progress bar
    tqdm_params = {
        'total': len(data),
        'desc': f"Processing {func.__name__.replace('_', ' ')}",
        'dynamic_ncols': True,
        'mininterval': 0.5,
        'maxinterval': 1.0,
        'miniters': 1,
        'file': sys.stdout,
        'disable': False
    }
    
    workers = workers or (multiprocessing.cpu_count() - 1 or 1)
    
    with multiprocessing.Pool(workers) as pool:
        results = []
        with tqdm(**tqdm_params) as pbar:
            for batch in (islice(data, i, i+batch_size) 
                         for i in range(0, len(data), batch_size)):
                batch_results = pool.map(func, batch)
                results.extend(batch_results)
                pbar.update(len(batch_results))
                sys.stdout.flush()  # Force display update
                
    return results

def get_system_stats():
    """
    Get comprehensive system stats.
    
    Returns:
        Dict with 'cpu' and 'mem' usage info
    """
    import psutil
    
    # CPU stats
    cpu_percent = psutil.cpu_percent(percpu=True)
    avg_cpu = sum(cpu_percent)/len(cpu_percent)
    
    # Memory stats
    mem = psutil.virtual_memory()
    
    return {
        'cpu': f"{avg_cpu:.1f}% ({max(cpu_percent):.1f}% peak)",
        'mem': f"{mem.used/1024/1024:.0f}/{mem.total/1024/1024:.0f}"
    }

def generate_structure_report(input_path: str, output_path: str, sample_size: int = 1000):
    """
    Generate detailed report of profile data structure.
    
    Args:
        input_path: Path to profile data
        output_path: Where to save report
        sample_size: Number of profiles to analyze
    """
    from collections import defaultdict
    
    stats = {
        'file': os.path.basename(input_path),
        'records_analyzed': 0,
        'max_depth': 0,
        'field_coverage': defaultdict(dict)
    }
    
    with open(input_path) as f:
        for i, line in enumerate(tqdm(f, total=sample_size, desc="Analyzing structure")):
            if i >= sample_size:
                break
                
            try:
                profile = json.loads(line)
                stats['records_analyzed'] += 1
                
                # Analyze field presence and types
                analyze_structure(profile, stats, current_path="")
                
            except json.JSONDecodeError:
                continue
    
    # Calculate percentages
    for field in stats['field_coverage']:
        stats['field_coverage'][field]['presence'] = (
            stats['field_coverage'][field].get('count', 0) / stats['records_analyzed'] * 100
        )
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def analyze_structure(data, stats, current_path, depth=0):
    """
    Recursively analyze nested data structure.
    
    Args:
        data: Input data to analyze
        stats: Accumulated stats
        current_path: Current path in data structure
        depth: Current recursion depth
    """
    stats['max_depth'] = max(stats['max_depth'], depth)
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}.{key}" if current_path else key
            
            # Update field stats
            if new_path not in stats['field_coverage']:
                stats['field_coverage'][new_path] = {
                    'count': 0,
                    'types': set(),
                    'examples': set()
                }
                
            stats['field_coverage'][new_path]['count'] += 1
            stats['field_coverage'][new_path]['types'].add(type(value).__name__)
            
            # Keep 3 example values
            if len(stats['field_coverage'][new_path]['examples']) < 3:
                if value and isinstance(value, (str, int, float, bool)):
                    stats['field_coverage'][new_path]['examples'].add(str(value)[:100])
            
            # Recurse into nested structures
            analyze_structure(value, stats, new_path, depth+1)
            
    elif isinstance(data, list) and data and isinstance(data[0], (dict, list)):
        analyze_structure(data[0], stats, current_path, depth+1)

def normalize_profiles(input_path: str, output_path: str):
    """
    Standardize profile field formats and values.
    
    Args:
        input_path: Path to input profiles
        output_path: Path for standardized output
    """
    from data_standardizer import DataStandardizer
    
    print(f"🔍 Loading {input_path}")
    with open(input_path) as f:
        profiles = json.load(f)
    
    print(f"⚙️ Normalizing {len(profiles)} profiles")
    standardizer = DataStandardizer(verbose=False)
    normalized = []
    for idx, profile in enumerate(profiles):
        try:
            result = standardizer.normalize_profile(profile)
            normalized.append(result)
        except Exception as e:
            print(f"Skipped profile {idx}: {str(e)}")
    
    print(f"💾 Saving {len(normalized)} normalized profiles")
    with open(output_path, 'w') as f:
        json.dump(normalized, f, indent=2)
    
    print(f"✅ Successfully saved standardized data to {output_path}")

def process_dataset(input_path: str, output_dir: str):
    """
    Process dataset through all pipeline steps sequentially.
    
    Args:
        input_path: Path to raw input data
        output_dir: Directory for all output files
    """
    # Existing sequential processing logic
    pass

if __name__ == "__main__":
    """
    Command line interface for running the pipeline.
    Usage: python data_cleaning.py [OPTIONS]
    
    Options:
      --input: Raw input file path
      --output: Base output directory
      --milvus: Milvus output path
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='LinkedIn data processing pipeline')
    parser.add_argument('--input', required=True, help='Input JSON path')
    parser.add_argument('--output', required=True, help='Base output directory')
    parser.add_argument('--milvus', help='Milvus output path')
    args = parser.parse_args()
    
    # Create output paths
    os.makedirs(args.output, exist_ok=True)
    filtered_path = os.path.join(args.output, "filtered_profiles.json")
    classified_path = os.path.join(args.output, "classified_profiles.json")
    cleaned_path = os.path.join(args.output, "cleaned_profiles.json")
    summarized_path = os.path.join(args.output, "summarized_profiles.json")
    
    start_time = time.time()
    
    # Run the full pipeline
    process_pipeline(
        input_path=args.input,
        filtered_output_path=filtered_path,
        classified_output_path=classified_path,
        cleaned_output_path=cleaned_path,
        summarized_output_path=summarized_path
    )
    
    print(f"\nPipeline completed in {time.time()-start_time:.1f}s")
    print("--- Output Files ---")
    print(f"Filtered: {filtered_path}")
    print(f"Classified: {classified_path}")
    print(f"Cleaned: {cleaned_path}")
    print(f"Summarized: {summarized_path}")
    if args.milvus:
        print(f"Milvus-ready: {args.milvus}")

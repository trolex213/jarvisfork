# LinkedIn Profile Processing Pipeline

This pipeline processes raw LinkedIn profile data for network analysis and storage in Milvus vector database.

## Optimization Note
- Uses ROCm-optimized PyTorch for accelerated embedding generation on AMD GPUs

## Files Overview

1. **data_cleaning.py** - Main pipeline script that:
   - Filters profiles by NYC region
   - Cleans and standardizes profile data
   - Classifies student profiles
   - Generates profile summaries
   - Prepares Milvus schema

2. **milvus_data_prep.py** - Processes cleaned profiles for Milvus by:
   - Generating embeddings using BAAI/bge-large-en-v1.5 model
   - Validating and transforming profile data
   - Calculating experience metrics
   - Creating optimized schema for hybrid search

## Usage

### 1. Run the full pipeline:
```bash
python data_cleaning.py \
  --input raw_profiles.json \
  --filtered filtered_profiles.json \
  --classified classified_profiles.json \
  --summarized summarized_profiles.json
```

### 2. Prepare Milvus data:
```bash
python milvus_data_prep.py \
  --input summarized_profiles.json \
  --output milvus_profiles.json \
  --batch_size 32 \
  --schema milvus_schema.json
```

## Output Files

- `filtered_profiles.json`: NYC region profiles
- `classified_profiles.json`: Profiles with student classification
- `summarized_profiles.json`: Condensed profile representations
- `milvus_profiles.json`: Final data ready for Milvus import
- `milvus_schema.json`: Collection schema definition

## Requirements
- Python 3.11+
- See requirements.txt for dependencies

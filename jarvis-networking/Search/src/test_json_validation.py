import json
import sys
from pathlib import Path

def validate_and_repair_json(file_path: str):
    """
    Validate JSON file and attempt repair if invalid
    Returns:
        tuple: (is_valid, record_count, repaired_path, repair_stats)
    """
    repair_stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'invalid_lines': 0,
        'line_errors': []
    }
    
    try:
        with open(file_path) as f:
            data = json.load(f)
        print(f" Valid JSON with {len(data)} records")
        return True, len(data), file_path, repair_stats
    except json.JSONDecodeError as e:
        print(f" Found invalid JSON: {str(e)}")
        print("Attempting repair...")
        
        repaired = []
        with open(file_path) as f:
            for i, line in enumerate(f, 1):
                repair_stats['total_lines'] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    repaired.append(json.loads(line))
                    repair_stats['valid_lines'] += 1
                except json.JSONDecodeError as le:
                    repair_stats['invalid_lines'] += 1
                    repair_stats['line_errors'].append({
                        'line_number': i,
                        'error': str(le),
                        'line_content': line[:100]  # First 100 chars for context
                    })
                    print(f"  Error in line {i}: {str(le)}")
                    
        if repaired:
            repaired_path = f"{Path(file_path).stem}_repaired.json"
            with open(repaired_path, 'w') as f:
                json.dump(repaired, f)
            print(f"  Repaired file saved to {repaired_path}")
            print(f"  Contains {len(repaired)} valid records")
            
            # Save repair stats to a separate file
            stats_path = f"{Path(file_path).stem}_repair_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(repair_stats, f, indent=2)
            print(f"  Repair statistics saved to {stats_path}")
            
            return False, len(repaired), repaired_path, repair_stats
        else:
            print("  Could not recover any valid JSON records")
            return False, 0, None, repair_stats

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_json_validation.py <json_file_path>")
        sys.exit(1)
    
    is_valid, count, repaired_path, repair_stats = validate_and_repair_json(sys.argv[1])
    if not is_valid and repaired_path:
        print(f"\nYou can use the repaired file:\n  {repaired_path}")
        print(f"\nRepair statistics:\n  Valid lines: {repair_stats['valid_lines']}")
        print(f"  Invalid lines: {repair_stats['invalid_lines']}")
        print(f"  Detailed stats saved to: {Path(sys.argv[1]).stem}_repair_stats.json")

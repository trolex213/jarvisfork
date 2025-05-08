"""
JSON Structure Analyzer and Metadata Optimizer
Creates optimal hybrid search structure for Milvus based on data analysis
"""
import json
from collections import defaultdict
from typing import Dict, Any, List
import argparse

class JSONStructureAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.field_stats = defaultdict(lambda: {"count": 0, "examples": set(), "types": set()})
        self.max_depth = 0
        self.total_records = 0

    def analyze(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Main analysis function that handles both NDJSON and standard JSON arrays"""
        with open(self.filepath) as f:
            try:
                # First try reading as standard JSON array
                data = json.load(f)
                if isinstance(data, list):
                    for i, record in enumerate(data):
                        if i >= sample_size:
                            break
                        self._analyze_record(record)
                        self.total_records += 1
                else:
                    # Handle single JSON object case
                    self._analyze_record(data)
                    self.total_records += 1
            except json.JSONDecodeError:
                # Fallback to NDJSON processing
                f.seek(0)
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    if line.strip():
                        self._analyze_record(json.loads(line))
                        self.total_records += 1
        return self._generate_report()

    def _analyze_record(self, record: Dict[str, Any], path: str = "", depth: int = 0):
        """Recursively analyze a JSON record"""
        self.max_depth = max(self.max_depth, depth)
        
        for key, value in record.items():
            current_path = f"{path}.{key}" if path else key
            self.field_stats[current_path]["count"] += 1
            self.field_stats[current_path]["types"].add(type(value).__name__)
            
            if len(self.field_stats[current_path]["examples"]) < 3:
                if isinstance(value, (str, int, float, bool)):
                    self.field_stats[current_path]["examples"].add(str(value)[:100])
            
            if isinstance(value, dict):
                self._analyze_record(value, current_path, depth + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                for item in value[:3]:
                    self._analyze_record(item, f"{current_path}[]", depth + 1)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report with recommendations"""
        field_coverage = {
            path: {
                "presence": (stats["count"] / self.total_records) * 100,
                "types": list(stats["types"]),
                "examples": list(stats["examples"])[:3]
            }
            for path, stats in self.field_stats.items()
        }
        
        search_fields = self._identify_search_fields(field_coverage)
        
        return {
            "file": self.filepath,
            "records_analyzed": self.total_records,
            "max_depth": self.max_depth,
            "field_coverage": field_coverage,
            "recommended_metadata_structure": self._recommend_structure(search_fields),
            "high_value_fields": search_fields
        }

    def _identify_search_fields(self, coverage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify fields most valuable for hybrid search"""
        SCORING_WEIGHTS = {
            "text": 1.5, "str": 1.2, "list": 0.8, "numeric": 0.5, "bool": 0.3
        }
        
        scored_fields = []
        for path, stats in coverage.items():
            type_score = max(SCORING_WEIGHTS.get(t.lower(), 0.5) for t in stats["types"])
            coverage_score = stats["presence"] / 100
            score = type_score * coverage_score
            
            if score > 0.5:
                scored_fields.append({
                    "path": path,
                    "score": round(score, 2),
                    "presence": stats["presence"],
                    "types": stats["types"]
                })
        
        return sorted(scored_fields, key=lambda x: -x["score"])

    def _recommend_structure(self, high_value_fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommended metadata structure"""
        structure = {
            "profile": {"description": "Core profile information", "fields": []},
            "professional": {"description": "Career-related information", "fields": []},
            "content": {"description": "Generated/extracted content", "fields": []},
            "system": {"description": "Processing metadata", "fields": []}
        }
        
        for field in high_value_fields:
            path = field["path"]
            if any(x in path for x in ["name", "title", "headline", "location"]):
                structure["profile"]["fields"].append(path)
            elif any(x in path for x in ["experience", "education", "skills", "projects"]):
                structure["professional"]["fields"].append(path)
            elif any(x in path for x in ["summary", "keywords", "interests", "text"]):
                structure["content"]["fields"].append(path)
            elif any(x in path for x in ["date", "version", "source", "id"]):
                structure["system"]["fields"].append(path)
        
        return structure

    def generate_milvus_schema(self, output_path: str):
        """Generate Milvus collection schema based on analysis"""
        schema = {
            "collection_name": "hybrid_network_graphs",
            "fields": [
                {"name": "id", "type": "VARCHAR", "is_primary": True},
                {"name": "vector", "type": "FLOAT_VECTOR", "dim": 1024},
                *[
                    {
                        "name": f"metadata_{key_path.replace('.', '_')}",
                        "type": self._determine_field_type(stats),
                        "params": {"index_type": "TRIE" if self._determine_field_type(stats) == "VARCHAR" else "STL_SORT"}
                    }
                    for key_path, stats in self.field_stats.items()
                    if stats["count"] > self.total_records * 0.1  # Only include fields present in >10% of records
                ]
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)

    def _determine_field_type(self, stats: Dict) -> str:
        if 'list' in stats['types']:
            return "FLOAT_VECTOR"
        if 'int' in stats['types']:
            return "INT64"
        if 'float' in stats['types']:
            return "FLOAT"
        return "VARCHAR"

class HybridSearchAnalyzer:
    """Analyzes JSON structures for optimal hybrid search key selection"""
    
    def __init__(self, min_cardinality=50, max_nested_depth=3):
        self.field_stats = defaultdict(lambda: {
            'count': 0,
            'types': set(),
            'nested_paths': defaultdict(int),
            'unique_values': set()
        })
        self.min_cardinality = min_cardinality
        self.max_nested_depth = max_nested_depth

    def analyze(self, data: Dict, current_path: str = '') -> None:
        """Recursively analyze JSON structure for hybrid search optimization"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                self._update_field_stats(new_path, value)
                if isinstance(value, dict) and len(new_path.split('.')) < self.max_nested_depth:
                    self.analyze(value, new_path)

    def _update_field_stats(self, path: str, value: Any) -> None:
        stats = self.field_stats[path]
        stats['count'] += 1
        stats['types'].add(type(value).__name__)
        
        if isinstance(value, (str, int, float, bool)):
            stats['unique_values'].add(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            stats['nested_paths'][path + '[]'] += 1

    def get_hybrid_keys(self) -> Dict[str, List[str]]:
        """Recommend keys based on field characteristics"""
        recommendations = {
            'primary_key_candidates': [],
            'filter_fields': [],
            'vectorizable_fields': [],
            'high_cardinality_fields': []
        }

        for path, stats in self.field_stats.items():
            unique_count = len(stats['unique_values'])
            data_types = stats['types']

            # Primary key detection
            if 'str' in data_types and unique_count / stats['count'] > 0.95:
                recommendations['primary_key_candidates'].append(path)

            # Filter field candidates
            if unique_count < self.min_cardinality and 'str' in data_types:
                recommendations['filter_fields'].append(f"{path}::string")
            elif 'float' in data_types or 'int' in data_types:
                recommendations['filter_fields'].append(f"{path}::{max(data_types, key=lambda x: x in ['float', 'int'])}")

            # Vector field candidates
            if 'list' in data_types and any(t in ['float', 'int'] for t in stats['types']):
                recommendations['vectorizable_fields'].append(path)

            if unique_count > self.min_cardinality * 10:
                recommendations['high_cardinality_fields'].append(path)

        return recommendations

    def generate_milvus_schema(self, output_path: str) -> None:
        """Generate recommended Milvus schema based on analysis"""
        schema = []
        for path, stats in self.field_stats.items():
            field_type = self._determine_field_type(stats)
            schema.append({
                "name": path.replace('.', '_'),
                "dtype": field_type,
                "params": {
                    "index_type": "TRIE" if field_type == "VARCHAR" else "STL_SORT",
                    "metric_type": "L2" if field_type == "FLOAT_VECTOR" else ""
                }
            })
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)

    def _determine_field_type(self, stats: Dict) -> str:
        if 'list' in stats['types']:
            return "FLOAT_VECTOR"
        if 'int' in stats['types']:
            return "INT64"
        if 'float' in stats['types']:
            return "FLOAT"
        return "VARCHAR"

def save_report(report: Dict[str, Any], output_file: str = "structure_report.json"):
    """Save analysis report to file"""
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze JSON structure for hybrid search optimization")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("--output", help="Output report file path", default="structure_report.json")
    parser.add_argument("--schema-output", help="Output schema file path", default="milvus_schema.json")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of records to analyze")
    args = parser.parse_args()
    
    analyzer = JSONStructureAnalyzer(args.input)
    report = analyzer.analyze(args.sample_size)
    
    hybrid_analyzer = HybridSearchAnalyzer()
    with open(args.input) as f:
        try:
            # First try reading as standard JSON array
            data = json.load(f)
            if isinstance(data, list):
                for i, record in enumerate(data):
                    if i >= args.sample_size:
                        break
                    hybrid_analyzer.analyze(record)
            else:
                # Handle single JSON object case
                hybrid_analyzer.analyze(data)
        except json.JSONDecodeError:
            # Fallback to NDJSON processing
            f.seek(0)
            for i, line in enumerate(f):
                if i >= args.sample_size:
                    break
                if line.strip():
                    hybrid_analyzer.analyze(json.loads(line))
    
    hybrid_keys = hybrid_analyzer.get_hybrid_keys()
    hybrid_analyzer.generate_milvus_schema(args.schema_output)
    
    report["hybrid_keys"] = hybrid_keys
    
    save_report(report, args.output)
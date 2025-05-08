"""
Enhanced LinkedIn Data Quality Analysis Tool
"""
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional

class LinkedInDataAnalyzer:
    LINKEDIN_ARTIFACTS = [
        'log in', 'login required', 'captcha', 'restricted',
        'profile unavailable', 'see more', 'show more'
    ]
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.report = {
            'total_records': 0,
            'field_coverage': {},
            'data_issues': [],
            'value_distributions': defaultdict(dict),
            'nested_stats': defaultdict(dict)
        }
    
    def analyze(self, sample_size: Optional[int] = 1000) -> Dict:
        """Run comprehensive analysis with visualization support"""
        data = self._load_data(sample_size)
        self.report['total_records'] = len(data)
        
        for record in data:
            self._analyze_record(record)
        
        self._calculate_stats()
        return self.report
    
    def _analyze_record(self, record: Dict) -> None:
        """Analyze a single record including nested structures"""
        for field, value in record.items():
            self._init_field_tracking(field)
            
            if self._is_empty(value):
                self.report['field_coverage'][field]['empty'] += 1
            else:
                self._check_artifacts(field, value)
                self._check_html(field, value)
                self._handle_nested(field, value)
                
            self.report['field_coverage'][field]['count'] += 1
    
    def _handle_nested(self, field: str, value: Any) -> None:
        """Recursively analyze nested structures"""
        if isinstance(value, dict):
            for k, v in value.items():
                nested_field = f"{field}.{k}"
                self._init_field_tracking(nested_field)
                
                if self._is_empty(v):
                    self.report['field_coverage'][nested_field]['empty'] += 1
                else:
                    self._check_artifacts(nested_field, v)
                    self._handle_nested(nested_field, v)  # Recurse
                
                self.report['field_coverage'][nested_field]['count'] += 1
                
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for i, item in enumerate(value):
                self._handle_nested(f"{field}[{i}]", item)
    
    def _check_artifacts(self, field: str, value: str) -> None:
        """Check for LinkedIn-specific scraping artifacts"""
        if isinstance(value, str):
            value_lower = value.lower()
            for artifact in self.LINKEDIN_ARTIFACTS:
                if artifact in value_lower:
                    self.report['data_issues'].append({
                        'type': 'linkedin_artifact',
                        'field': field,
                        'sample': value[:200]
                    })
    
    def _check_html(self, field: str, value: str) -> None:
        """Detect HTML fragments in string fields"""
        if isinstance(value, str) and any(tag in value for tag in ['<div', '<span', 'class="']):
            self.report['data_issues'].append({
                'type': 'html_fragment', 
                'field': field,
                'sample': value[:200]
            })
    
    def _is_empty(self, value: Any) -> bool:
        """Check for empty/null values"""
        return value in (None, "", [], {})
    
    def _init_field_tracking(self, field: str) -> None:
        """Initialize tracking for a field"""
        if field not in self.report['field_coverage']:
            self.report['field_coverage'][field] = {'count': 0, 'empty': 0}
    
    def _calculate_stats(self) -> None:
        """Calculate completion percentages"""
        for field in self.report['field_coverage']:
            total = self.report['field_coverage'][field]['count']
            empty = self.report['field_coverage'][field].get('empty', 0)
            self.report['field_coverage'][field]['completion_pct'] = (total - empty) / total * 100
    
    def visualize(self, output_prefix: str = "dq_report") -> None:
        """Generate visualizations of data quality"""
        # Field completion heatmap
        df = pd.DataFrame.from_dict(self.report['field_coverage'], orient='index')
        df['completion_pct'].sort_values().plot(kind='barh', 
                                              title='Field Completion Rates')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_completion.png")
        plt.close()
        
        # Issues by type
        if self.report['data_issues']:
            issues_df = pd.DataFrame(self.report['data_issues'])
            issues_df['type'].value_counts().plot(kind='pie', 
                                                title='Data Issue Types')
            plt.savefig(f"{output_prefix}_issues.png")
            plt.close()
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive report"""
        report = [
            "=== LinkedIn Data Quality Report ===",
            f"File: {self.filepath}",
            f"Records Analyzed: {self.report['total_records']}",
            "\n--- Field Coverage ---"
        ]
        
        # Sort fields by completion percentage
        sorted_fields = sorted(
            self.report['field_coverage'].items(),
            key=lambda x: x[1]['completion_pct'],
            reverse=True
        )
        
        for field, stats in sorted_fields:
            report.append(
                f"{field.ljust(30)}: {stats['completion_pct']:.1f}% complete "
                f"(empty: {stats['empty']}/{stats['count']})"
            )
        
        # Add issue summary
        if self.report['data_issues']:
            report.extend(["\n--- Data Issues ---", 
                          f"Total issues: {len(self.report['data_issues'])}"])
            
            # Group issues by type
            issues_by_type = defaultdict(list)
            for issue in self.report['data_issues']:
                issues_by_type[issue['type']].append(issue)
            
            for issue_type, issues in issues_by_type.items():
                report.append(f"\n{issue_type.upper()} ({len(issues)}):")
                for issue in issues[:3]:  # Show 3 samples per type
                    report.append(f"  - {issue['field']}: {issue['sample']}")
        
        report_content = "\n".join(report)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content

    def _load_data(self, sample_size: Optional[int]) -> List[Dict]:
        """Load data with error handling and format detection"""
        data = []
        with open(self.filepath) as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                try:
                    record = json.loads(line)
                    # Convert lists of records to individual dicts
                    if isinstance(record, list):
                        data.extend([r for r in record if isinstance(r, dict)])
                    elif isinstance(record, dict):
                        data.append(record)
                except json.JSONDecodeError:
                    self.report['data_issues'].append({
                        'type': 'invalid_json',
                        'field': 'N/A',
                        'sample': line[:200] + ('...' if len(line) > 200 else '')
                    })
        return data

    def clean_data(self, sample_size: Optional[int] = None, new_filepath: Optional[str] = None) -> Dict:
        """Clean data and optionally save to new file"""
        cleaned_data = []
        changes_made = 0
        
        for record in self._load_data(sample_size):
            cleaned_record = record.copy()
            
            # Cleaning logic remains the same
            for field, value in record.items():
                if isinstance(value, str) and any(tag in value for tag in ['<div', '<span', 'class="']):
                    cleaned_record[field] = self._remove_html(value)
                    changes_made += 1
                
                if isinstance(value, str) and any(
                    artifact in value.lower() 
                    for artifact in self.LINKEDIN_ARTIFACTS
                ):
                    cleaned_record[field] = None
                    changes_made += 1
            
            cleaned_data.append(cleaned_record)
        
        if new_filepath:
            with open(new_filepath, 'w') as f:
                for record in cleaned_data:
                    f.write(json.dumps(record) + '\n')
        
        return {
            'cleaned_data': cleaned_data,
            'changes_made': changes_made,
            'original_record_count': len(cleaned_data)
        }
    
    def _remove_html(self, text: str) -> str:
        """Basic HTML tag removal"""
        import re
        return re.sub(r'<[^>]+>', '', text)
    
    def analyze_demographics(self) -> Dict:
        """Extract professional demographic insights"""
        data = self._load_data(sample_size=10000)  # Analyze first 10k records
        insights = {
            'top_companies': defaultdict(int),
            'common_degrees': defaultdict(int),
            'experience_distribution': [],
            'education_distribution': defaultdict(int)
        }
        
        for record in data:
            # Company analysis
            if 'current_company' in record and 'name' in record['current_company']:
                insights['top_companies'][record['current_company']['name']] += 1
            
            # Education analysis
            if 'education' in record and isinstance(record['education'], list):
                for edu in record['education']:
                    if 'degree' in edu:
                        insights['common_degrees'][edu['degree']] += 1
                    if 'field' in edu:
                        insights['education_distribution'][edu['field']] += 1
            
            # Experience analysis
            if 'experience' in record and isinstance(record['experience'], list):
                insights['experience_distribution'].append(len(record['experience']))
        
        # Process distributions
        insights['top_10_companies'] = dict(
            sorted(insights['top_companies'].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        insights['top_10_degrees'] = dict(
            sorted(insights['common_degrees'].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        insights['avg_experience_items'] = (
            sum(insights['experience_distribution']) / len(insights['experience_distribution'])
            if insights['experience_distribution'] else 0
        )
        
        return insights

# Example test cases for LinkedIn data
TEST_CASES = {
    "empty_profile": {"about": "", "experience": []},
    "artifact_profile": {"about": "Please log in to view this profile"},
    "html_profile": {"summary": "<div class=\"profile\">About me</div>"},
    "nested_data": {"education": [{"degree": "MBA", "school": {"name": "Harvard", "url": None}}]}
}

def run_tests():
    """Validate analyzer with test cases"""
    analyzer = LinkedInDataAnalyzer("test_data.json")
    
    for name, test_case in TEST_CASES.items():
        print(f"\nRunning test: {name}")
        analyzer._analyze_record(test_case)
    
    print("\nTest Results:")
    print(analyzer.generate_report())

if __name__ == "__main__":
    analyzer = LinkedInDataAnalyzer("summarized_processed.json")
    report = analyzer.analyze(sample_size=1000)
    print(analyzer.generate_report("quality_report.txt"))
    analyzer.visualize()
    
    # Uncomment to run test cases
    # run_tests()

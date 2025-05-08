import pandas as pd
import re
import numpy as np
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from collections import Counter

def normalize_company_names(input_csv, output_csv):
    """
    Normalize company names in a CSV file for optimized matching in cold outreach platform.
    
    Parameters:
    input_csv (str): Path to input CSV file with company names
    output_csv (str): Path to save the normalized CSV file
    """
    print("Loading company names dataset...")
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_csv, header=None)
        # Rename columns for clarity
        if len(df.columns) >= 2:
            df.columns = ['company_name', 'company_id'] + [f'col_{i}' for i in range(2, len(df.columns))]
        else:
            df.columns = ['company_name']
            df['company_id'] = np.arange(len(df))
        
        print(f"Loaded {len(df)} company names.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Store original data for before/after analysis
    before_data = {
        'lengths': df['company_name'].astype(str).str.len(),
        'word_counts': df['company_name'].astype(str).str.split().apply(len),
        'suffix_inc': df['company_name'].astype(str).str.contains(r'\bInc\.?(\s|$)', case=True, regex=True),
        'suffix_llc': df['company_name'].astype(str).str.contains(r'\bLLC\.?(\s|$)', case=True, regex=True),
        'suffix_corp': df['company_name'].astype(str).str.contains(r'\bCorp\.?(\s|$)', case=True, regex=True),
        'suffix_ltd': df['company_name'].astype(str).str.contains(r'\bLtd\.?(\s|$)', case=True, regex=True),
        'has_punct': df['company_name'].astype(str).str.contains(r'[^\w\s]')
    }
    
    # Create a copy of the original company name
    df['original_name'] = df['company_name']
    
    # Collect data for before/after analysis
    before_analysis = {
        'name_lengths': df['company_name'].astype(str).str.len(),
        'word_counts': df['company_name'].astype(str).str.split().apply(len),
        'capitalization': df['company_name'].astype(str).apply(lambda x: 'Mixed' if x.isupper() else ('Title' if x.istitle() else 'Other')),
        'contains_punctuation': df['company_name'].astype(str).str.contains(r'[^\w\s]'),
        'common_suffixes': {
            'Inc': df['company_name'].astype(str).str.contains(r'\bInc\.?(\s|$)', case=True, regex=True),
            'Corp': df['company_name'].astype(str).str.contains(r'\bCorp\.?(\s|$)', case=True, regex=True),
            'LLC': df['company_name'].astype(str).str.contains(r'\bLLC\.?(\s|$)', case=True, regex=True),
            'Ltd': df['company_name'].astype(str).str.contains(r'\bLtd\.?(\s|$)', case=True, regex=True),
        }
    }
    
    print("Normalizing company names...")
    
    # Standard text normalization
    df['normalized_name'] = df['company_name'].astype(str).str.strip()
    
    # 1. Convert to lowercase
    df['normalized_name'] = df['normalized_name'].str.lower()
    
    # 2. Replace multiple spaces with a single space
    df['normalized_name'] = df['normalized_name'].str.replace(r'\s+', ' ', regex=True)
    
    # 3. Standardize legal entity designations
    legal_entity_patterns = {
        r'\bincorporated\b': 'inc',
        r'\binc\.?': 'inc',
        r'\bcorporation\b': 'corp',
        r'\bcorp\.?': 'corp',
        r'\bcompany\b': 'co',
        r'\bco\.?': 'co',
        r'\blimited\b': 'ltd',
        r'\bltd\.?': 'ltd',
        r'\blimited liability company\b': 'llc',
        r'\bllc\.?': 'llc',
        r'\bllp\.?': 'llp',
        r'\blimited liability partnership\b': 'llp',
        r',': '',  # Remove commas
    }
    
    for pattern, replacement in legal_entity_patterns.items():
        df['normalized_name'] = df['normalized_name'].str.replace(pattern, replacement, regex=True)
    
    # 4. Remove common symbols and punctuation
    df['normalized_name'] = df['normalized_name'].str.replace(r'[^\w\s]', '', regex=True)
    
    # 5. Standardize common abbreviations
    common_abbr = {
        r'\bintl\b': 'international',
        r'\bint\'?l\b': 'international',
        r'\bmfg\b': 'manufacturing',
        r'\bmgt\b': 'management',
        r'\bsvcs\b': 'services',
        r'\bsvc\b': 'service',
        r'\btech\b': 'technology',
        r'\btechs\b': 'technologies',
        r'\btechnologies\b': 'technology',
        r'\bsys\b': 'systems',
        r'\bgrp\b': 'group',
        r'\bassoc\b': 'associates',
        r'\bassn\b': 'association',
        r'\bent\b': 'enterprises',
        r'\bdept\b': 'department',
    }
    
    for pattern, replacement in common_abbr.items():
        df['normalized_name'] = df['normalized_name'].str.replace(pattern, replacement, regex=True)
    
    # 6. Create a more aggressive normalization for fuzzy matching
    df['fuzzy_name'] = df['normalized_name'].copy()
    
    # Remove common filler words for fuzzy matching
    filler_words = [
        r'\bthe\b', r'\band\b', r'\bof\b', r'\bin\b', r'\bat\b', r'\bfor\b', 
        r'\bto\b', r'\ba\b', r'\ban\b', r'\bon\b', r'\bwith\b', r'\bas\b'
    ]
    
    for word in filler_words:
        df['fuzzy_name'] = df['fuzzy_name'].str.replace(word, '', regex=True)
    
    # Remove all legal suffixes for fuzzy matching
    legal_suffixes = [
        r'\binc\b', r'\bcorp\b', r'\bco\b', r'\bltd\b', r'\bllc\b', 
        r'\bllp\b', r'\bplc\b', r'\bgmbh\b', r'\bag\b'
    ]
    
    for suffix in legal_suffixes:
        df['fuzzy_name'] = df['fuzzy_name'].str.replace(suffix, '', regex=True)
    
    # Clean up extra spaces from removals
    df['fuzzy_name'] = df['fuzzy_name'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 7. Generate domain-friendly name (for potential email matching)
    df['domain_name'] = df['fuzzy_name'].str.replace(' ', '').str.replace(r'[^\w]', '', regex=True)
    
    # 8. Extract industry keywords if possible (simplified version)
    common_industries = [
        'technology', 'tech', 'software', 'healthcare', 'health', 'financial', 
        'finance', 'bank', 'insurance', 'retail', 'manufacturing', 'consulting', 
        'media', 'pharmaceutical', 'pharma', 'energy', 'telecom', 'education', 
        'automotive', 'auto', 'real estate', 'construction', 'food', 'hospitality'
    ]
    
    df['industry_hint'] = ''
    
    for industry in common_industries:
        mask = df['normalized_name'].str.contains(r'\b' + industry + r'\b', regex=True)
        df.loc[mask, 'industry_hint'] = df.loc[mask, 'industry_hint'].str.cat([industry] * mask.sum(), sep=', ')
    
    # Remove leading comma if exists
    df['industry_hint'] = df['industry_hint'].str.strip().str.strip(',').str.strip()
    
    # Generate name_tokens
    df['name_tokens'] = df['normalized_name'].apply(
        lambda x: [word for word in x.split() if word not in [
            'inc', 'corp', 'co', 'ltd', 'llc', 'llp', 'plc', 'gmbh', 'ag',
            'the', 'and', 'of', 'for', 'in', 'at', 'by', 'with'
        ]]
    )
    
    # Generate abbreviations
    def create_abbreviation(name):
        # Split into words
        words = name.split()
        # Remove common legal suffixes and filler words
        filtered_words = [word for word in words if word not in [
            'inc', 'corp', 'co', 'ltd', 'llc', 'llp', 'plc', 'gmbh', 'ag',
            'the', 'and', 'of', 'for', 'in', 'at', 'by', 'with'
        ]]
        
        # If no words remain after filtering, use original words
        if not filtered_words and words:
            filtered_words = words
            
        # Create abbreviation from first letters
        if filtered_words:
            abbr = ''.join(word[0] for word in filtered_words)
            return abbr
        return ""
    
    df['abbreviation'] = df['normalized_name'].apply(create_abbreviation)
    
    # Generate name variants
    def create_name_variants(row):
        variants = []
        
        # Original normalized name
        normalized = row['normalized_name']
        variants.append(normalized)
        
        # Without legal suffix
        for suffix in ['inc', 'corp', 'co', 'ltd', 'llc', 'llp', 'plc', 'gmbh', 'ag']:
            if normalized.endswith(f" {suffix}"):
                variants.append(normalized[:-len(suffix)-1].strip())
                break
        
        # Add fuzzy name if different
        fuzzy = row['fuzzy_name']
        if fuzzy not in variants:
            variants.append(fuzzy)
            
        # Add abbreviation if meaningful
        abbr = row['abbreviation']
        if len(abbr) >= 2 and abbr not in variants:
            variants.append(abbr)
            
        # First token only (if multiple tokens exist)
        tokens = row['name_tokens']
        if len(tokens) > 1 and tokens[0] not in variants:
            variants.append(tokens[0])
            
        return variants
    
    df['name_variants'] = df.apply(create_name_variants, axis=1)
    
    # Collect data for after normalization analysis
    after_analysis = {
        'name_lengths': df['normalized_name'].str.len(),
        'word_counts': df['normalized_name'].str.split().apply(len),
        'token_counts': df['name_tokens'].apply(len),
        'variant_counts': df['name_variants'].apply(len),
        'abbr_lengths': df['abbreviation'].str.len(),
        'has_industry': df['industry_hint'].str.len() > 0,
    }
    
    print("Normalization complete. Saving results...")
    
    # Create output dataset with useful columns for matching algorithm
    output_df = df[['company_id', 'original_name', 'normalized_name', 'fuzzy_name', 
                    'domain_name', 'industry_hint', 'name_tokens', 'abbreviation', 'name_variants']]
    
    output_df.to_csv(output_csv, index=False)
    print(f"Normalized company names saved to {output_csv}")
    
    # Collect after normalization data for analysis
    after_data = {
        'lengths': df['normalized_name'].str.len(),
        'word_counts': df['normalized_name'].str.split().apply(len),
        'token_counts': df['name_tokens'].apply(len),
        'variant_counts': df['name_variants'].apply(len),
        'has_industry': df['industry_hint'].str.len() > 0,
        'abbrev_lengths': df['abbreviation'].str.len()
    }
    
    # Display sample of normalized data
    print("\nSample of normalized data:")
    sample_df = output_df.head()
    
    # Make the sample more readable for display
    for col in ['name_tokens', 'name_variants']:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].apply(lambda x: str(x))
    
    print(sample_df)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(f"Total companies: {len(output_df)}")
    
    # Name token statistics
    if 'name_tokens' in output_df.columns:
        token_lengths = output_df['name_tokens'].apply(len)
        print(f"Average tokens per company name: {token_lengths.mean():.1f}")
        print(f"Max tokens in a company name: {token_lengths.max()}")
    
    # Name variant statistics
    if 'name_variants' in output_df.columns:
        variant_counts = output_df['name_variants'].apply(len)
        print(f"Average variants per company: {variant_counts.mean():.1f}")
        print(f"Max variants for a company: {variant_counts.max()}")
    
    # Industry distribution (if found)
    if output_df['industry_hint'].str.strip().any():
        print("\nIndustry distribution (from detected keywords):")
        industry_breakdown = {}
        for industry in common_industries:
            count = output_df[output_df['industry_hint'].str.contains(industry, na=False)].shape[0]
            if count > 0:
                industry_breakdown[industry] = count
        
        for industry, count in sorted(industry_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"  {industry}: {count} companies ({count/len(output_df)*100:.1f}%)")
    
    # Generate histogram visualizations
    generate_distribution_histograms(before_data, after_data, len(df))
    
    return output_df

def generate_distribution_histograms(before, after, total_count):
    """Generate histograms showing distributions before and after normalization"""
    plt.figure(figsize=(20, 15))
    
    # 1. Name Length Distribution
    plt.subplot(3, 2, 1)
    max_length = max(before['lengths'].max(), after['lengths'].max())
    bins = range(0, int(max_length) + 5, 5)  # Group by 5 characters
    
    plt.hist(before['lengths'], bins=bins, alpha=0.7, label='Before Normalization', color='blue')
    plt.hist(after['lengths'], bins=bins, alpha=0.7, label='After Normalization', color='green')
    
    plt.title('Company Name Length Distribution', fontsize=14)
    plt.xlabel('Character Length', fontsize=12)
    plt.ylabel('Number of Companies', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate and display statistics on the plot
    plt.text(0.7, 0.9, f"Before: avg={before['lengths'].mean():.1f} chars\nAfter: avg={after['lengths'].mean():.1f} chars",
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 2. Word Count Distribution
    plt.subplot(3, 2, 2)
    max_words = max(before['word_counts'].max(), after['word_counts'].max())
    
    plt.hist(before['word_counts'], bins=range(1, int(max_words) + 2), alpha=0.7, label='Before Normalization', color='blue')
    plt.hist(after['word_counts'], bins=range(1, int(max_words) + 2), alpha=0.7, label='After Normalization', color='green')
    
    plt.title('Word Count Distribution', fontsize=14)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Number of Companies', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(1, int(max_words) + 1))
    
    # Calculate and display statistics on the plot
    plt.text(0.7, 0.9, f"Before: avg={before['word_counts'].mean():.1f} words\nAfter: avg={after['word_counts'].mean():.1f} words",
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 3. Legal Suffix Distribution (Before)
    plt.subplot(3, 2, 3)
    suffix_labels = ['Inc', 'LLC', 'Corp', 'Ltd', 'None']
    
    # Count each suffix
    suffix_counts = [
        before['suffix_inc'].sum(),
        before['suffix_llc'].sum(),
        before['suffix_corp'].sum(),
        before['suffix_ltd'].sum(),
        total_count - (before['suffix_inc'].sum() + before['suffix_llc'].sum() + 
                      before['suffix_corp'].sum() + before['suffix_ltd'].sum())
    ]
    
    # Color-code by type
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Calculate percentages
    suffix_pcts = [count / total_count * 100 for count in suffix_counts]
    
    # Create bar chart with percentage labels
    bars = plt.bar(suffix_labels, suffix_pcts, color=colors)
    plt.title('Legal Suffix Distribution (Before Normalization)', fontsize=14)
    plt.xlabel('Suffix Type', fontsize=12)
    plt.ylabel('Percentage of Companies (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{suffix_pcts[i]:.1f}%', ha='center', fontsize=10)
    
    # 4. Company Name Features
    plt.subplot(3, 2, 4)
    feature_labels = ['Has Punctuation', '2+ Tokens', '3+ Name Variants', 'Has Industry Hint', '2+ Char Abbreviation']
    
    # Count each feature
    feature_counts = [
        before['has_punct'].sum(),
        (after['token_counts'] >= 2).sum(),
        (after['variant_counts'] >= 3).sum(),
        after['has_industry'].sum(),
        (after['abbrev_lengths'] >= 2).sum()
    ]
    
    # Calculate percentages
    feature_pcts = [count / total_count * 100 for count in feature_counts]
    
    # Create bar chart with percentage labels
    colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bars = plt.bar(feature_labels, feature_pcts, color=colors)
    plt.title('Company Name Features', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Percentage of Companies (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{feature_pcts[i]:.1f}%', ha='center', fontsize=10)
    
    # 5. Token Count Distribution
    plt.subplot(3, 2, 5)
    max_tokens = min(after['token_counts'].max(), 10)  # Cap at 10 for readability
    
    token_counts = after['token_counts'].value_counts().sort_index()
    
    # Filter to tokens 0-10 and calculate percentages
    token_counts = token_counts.loc[token_counts.index <= max_tokens]
    token_pcts = token_counts / total_count * 100
    
    # Create bar chart with percentage labels
    bars = plt.bar(token_counts.index, token_pcts, color='#1f77b4')
    plt.title('Meaningful Token Count Distribution (After Normalization)', fontsize=14)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Percentage of Companies (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(0, int(max_tokens) + 1))
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{token_pcts.iloc[i]:.1f}%', ha='center', fontsize=10)
    
    # 6. Name Variants Distribution
    plt.subplot(3, 2, 6)
    max_variants = min(after['variant_counts'].max(), 10)  # Cap at 10 for readability
    
    variant_counts = after['variant_counts'].value_counts().sort_index()
    
    # Filter to variants 0-10 and calculate percentages
    variant_counts = variant_counts.loc[variant_counts.index <= max_variants]
    variant_pcts = variant_counts / total_count * 100
    
    # Create bar chart with percentage labels
    bars = plt.bar(variant_counts.index, variant_pcts, color='#ff7f0e')
    plt.title('Name Variants Distribution (After Normalization)', fontsize=14)
    plt.xlabel('Number of Variants', fontsize=12)
    plt.ylabel('Percentage of Companies (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(range(0, int(max_variants) + 1))
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{variant_pcts.iloc[i]:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('company_name_distributions.png', dpi=300)
    plt.close()
    
    print("\nDistribution histograms saved to company_name_distributions.png")
    print("This visualization shows the distribution of company name characteristics before and after normalization.")
    
    # Generate before and after analysis
    generate_before_after_analysis(before_analysis, after_analysis, len(df))
    
    return df, output_df

def generate_before_after_analysis(before, after, total_count):
    """Generate and display before/after normalization statistics"""
    print("\n==== BEFORE AND AFTER NORMALIZATION ANALYSIS ====")
    
    # 1. Name length distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(before['name_lengths'], bins=30, alpha=0.5, label='Before')
    plt.hist(after['name_lengths'], bins=30, alpha=0.5, label='After')
    plt.title('Company Name Length Distribution')
    plt.xlabel('Character Length')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 2. Word count distribution
    plt.subplot(2, 2, 2)
    plt.hist(before['word_counts'], bins=range(1, 15), alpha=0.5, label='Before')
    plt.hist(after['word_counts'], bins=range(1, 15), alpha=0.5, label='After')
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. Legal suffix distribution
    plt.subplot(2, 2, 3)
    suffix_labels = list(before['common_suffixes'].keys())
    suffix_counts = [before['common_suffixes'][suffix].sum() for suffix in suffix_labels]
    
    # Calculate percentages
    suffix_percentages = [count / total_count * 100 for count in suffix_counts]
    
    plt.bar(suffix_labels, suffix_percentages)
    plt.title('Legal Suffix Distribution (Before)')
    plt.xlabel('Suffix Type')
    plt.ylabel('Percentage of Companies (%)')
    
    # 4. After normalization extras
    plt.subplot(2, 2, 4)
    extra_labels = ['Has Industry', '2+ Tokens', '3+ Variants', '2+ Char Abbrev']
    extra_counts = [
        after['has_industry'].sum(),
        (after['token_counts'] >= 2).sum(),
        (after['variant_counts'] >= 3).sum(),
        (after['abbr_lengths'] >= 2).sum()
    ]
    
    # Calculate percentages
    extra_percentages = [count / total_count * 100 for count in extra_counts]
    
    plt.bar(extra_labels, extra_percentages)
    plt.title('Normalization Enhancements')
    plt.xlabel('Feature')
    plt.ylabel('Percentage of Companies (%)')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig('normalization_analysis.png')
    plt.close()
    
    print("Analysis charts saved to normalization_analysis.png")
    
    # Text summary of the before and after changes
    print("\nBEFORE NORMALIZATION:")
    print(f"- Average name length: {before['name_lengths'].mean():.1f} characters")
    print(f"- Average word count: {before['word_counts'].mean():.1f} words")
    print(f"- Companies with punctuation: {before['contains_punctuation'].sum()} ({before['contains_punctuation'].sum()/total_count*100:.1f}%)")
    
    cap_counts = before['capitalization'].value_counts()
    for cap_type, count in cap_counts.items():
        print(f"- {cap_type} capitalization: {count} ({count/total_count*100:.1f}%)")
    
    for suffix, mask in before['common_suffixes'].items():
        print(f"- With '{suffix}' suffix: {mask.sum()} ({mask.sum()/total_count*100:.1f}%)")
    
    print("\nAFTER NORMALIZATION:")
    print(f"- Average name length: {after['name_lengths'].mean():.1f} characters")
    print(f"- Average word count: {after['word_counts'].mean():.1f} words")
    print(f"- Average meaningful tokens: {after['token_counts'].mean():.1f}")
    print(f"- Average name variants: {after['variant_counts'].mean():.1f}")
    print(f"- Companies with industry hints: {after['has_industry'].sum()} ({after['has_industry'].sum()/total_count*100:.1f}%)")
    
    # Word frequency analysis (before and after)
    before_words = []
    for name in before['name_lengths'].index:
        name_str = str(df.loc[name, 'company_name']).lower()
        words = re.findall(r'\b[a-z]+\b', name_str)
        before_words.extend(words)
    
    after_words = []
    for name in after['name_lengths'].index:
        name_str = str(df.loc[name, 'normalized_name'])
        words = re.findall(r'\b[a-z]+\b', name_str)
        after_words.extend(words)
    
    before_word_counts = Counter(before_words).most_common(10)
    after_word_counts = Counter(after_words).most_common(10)
    
    print("\nTop 10 most common words BEFORE normalization:")
    for word, count in before_word_counts:
        print(f"- '{word}': {count} occurrences")
    
    print("\nTop 10 most common words AFTER normalization:")
    for word, count in after_word_counts:
        print(f"- '{word}': {count} occurrences")

# Example usage
if __name__ == "__main__":
    input_file = "summarized_profiles.json"
    output_file = "normalized_company_names.csv"
    df, output_df = normalize_company_names(input_file, output_file)
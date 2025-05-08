"""
Optimized with Google Search API
"""
import time
import json
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
from googlesearch import search
import pandas as pd
import csv
import os
import argparse

# Constants with progress tracking
PROGRESS_FILE = Path("./progress.json")
RESULTS_FILE = Path("./results.json")
CACHE_DIR = Path("./.gemma3_cache")
BATCH_SIZE = 100  # Increased from 50
MAX_WORKERS = 8   # Increased from 4
SEARCH_RESULTS = 2  # Reduced from 3
REQUEST_TIMEOUT = 30  # Seconds

# Only Google Search rate limit (1 call per second)
SEARCH_RATE_LIMIT = 1.0

VALID_CATEGORIES = [
    'Administrative Services', 'Advertising', 'Agriculture and Farming', 'Apps',
    'Artificial Intelligence', 'Biotechnology', 'Blockchain & Cryptocurrency',
    'Clothing and Apparel', 'Commerce and Shopping', 'Community and Lifestyle',
    'Consumer Electronics', 'Consumer Goods', 'Content and Publishing',
    'Data and Analytics', 'Design', 'Education', 'Energy', 'Events',
    'Financial Services', 'Food and Beverage', 'Gaming', 'Government and Military',
    'Hardware', 'Health Care', 'Information Technology', 'Internet Services',
    'Lending and Investments', 'Manufacturing', 'Media and Entertainment',
    'Messaging and Telecommunications', 'Mobile', 'Music and Audio',
    'Natural Resources', 'Navigation and Mapping', 'Other', 'Payments',
    'Platforms', 'Privacy and Security', 'Professional Services', 'Real Estate',
    'Sales and Marketing', 'Science and Engineering', 'Software', 'Sports',
    'Sustainability', 'Transportation', 'Travel and Tourism', 'Video'
]

class CompanyCategorizerGemma:
    def __init__(self, output_file='company_categories_gemma.json', raw_file='company_categories_raw.json', batch_size=30):
        """Initialize with Google Search API"""
        self.model = 'gemma3:12b'
        self.enable_search = True
        
        # Fixed batch size
        self.min_batch_size = batch_size
        self.max_batch_size = batch_size
        self.current_batch_size = batch_size
        
        self.output_file = output_file
        self.raw_file = raw_file
        CACHE_DIR.mkdir(exist_ok=True)
        self.search_cache = {}

    def _rate_limit(self, last_call_time, interval):
        """Enforce minimum time between calls"""
        elapsed = time.time() - last_call_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        return time.time()

    def web_search(self, query):
        """Google Search with robust error handling"""
        if query in self.search_cache:
            return self.search_cache[query]
            
        max_retries = 2
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                self._last_search_time = self._rate_limit(self._last_search_time, SEARCH_RATE_LIMIT)
                results = list(search(query, num_results=SEARCH_RESULTS))
                
                if results:
                    result_text = "\n".join(results)
                    self.search_cache[query] = result_text
                    return result_text
                return ""
                
            except Exception as e:
                delay = base_delay * (2 ** attempt)
                print(f"Search error (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(delay)
                
        return ""

    def _verify_category(self, category):
        """Robust category verification with normalization"""
        if not category or not isinstance(category, str):
            return False
            
        # Normalize for comparison
        normalized = category.strip().lower()
        valid_categories = [c.lower() for c in VALID_CATEGORIES]
        
        # Check direct match or contains (e.g. "IT" matches "Information Technology")
        is_valid = (normalized in valid_categories) or \
                  any(normalized in valid_cat for valid_cat in valid_categories)
                  
        print(f"[VERIFICATION] Category '{category}' valid: {is_valid}")
        return is_valid

    def make_batch_request(self, company_names):
        """Process multiple companies in a single request"""
        try:
            prompt = f"""Classify these companies into exactly one of these categories:
            
            Categories : Definitions:
            
            Administrative Services :	Companies that are primarily engaged in providing a range of day-to-day office administrative services, such as financial planning; billing and record keeping; and physical distribution and logistics, for others on a contract or fee basis.
            Advertising :	Companies that create and manage the connection between companies, products, and consumers, translating their clients' messages into effective campaigns.
            Agriculture and Farming :	Companies that encompass all aspects of food production.
            Apps :	Companies that build applications.
            Artificial  Intelligence :	Companies that concern themselves with the simulation of human intelligence in machines.
            Biotechnology :	The broad area of biology involving living systems and organisms to develop or make products, or any technological application that uses biological systems, living organisms, or derivatives thereof, to make or modify products or processes for specific use.
            Blockchain & Cryptocurrency :	Companies that relate to decentralized digital currencies or the ledgers on which decentralized financial transactions are recorded.
            Clothing and Apparel :	Companies that design and sell clothing, footwear, and accessories.
            Commerce and Shopping :	Companies involved in the buying and selling of goods by consumers and/or businesses.
            Community and Lifestyle :	Companies that are set up and run by its founders primarily with the aim of sustaining a particular level of income and no more; or to provide a foundation from which to enjoy a particular lifestyle.
            Consumer Electronics :	Companies that produce electronic equipment intended for everyday use, typically in private homes.
            Consumer Goods	: Companies that relate to items purchased by individuals and households rather than by manufacturers and industries.
            Content and Publishing :	Companies that are involved in the creation of and distribution of content.
            Data and Analytics	: Companies that analyze raw data in order to make conclusions about that information.
            Design	: Companies that influence the experience a user has with all of a company's touch points.
            Education	: Companies that provide instruction and training in a wide variety of subjects.
            Energy	: Companies who concern themselves with researching and creating new forms of energy.
            Events	: Companies that have experience in the intricacies and creativity of throwing a memorable and rewarding corporate or social event.
            Financial Services	: Companies that provide a broad range of businesses that manage money.
            Food and Beverage	: Companies involved in processing raw food materials, packaging, and distributing them.
            Gaming	: Companies that involve the development, marketing, and monetization of video games.
            Government and Military	: Companies that are involved with, make things for, and deal with the government and/or military.
            Hardware	: Companies concerned with a physical component of any computer or telecommunications system.
            Health Care	: Companies within the economic system that provide goods and services to treat patients with curative, preventive, rehabilitative, and palliative care.
            Information Technology	: Companies that deal with computing, including hardware, software, telecommunications and generally anything involved in the transmittal of information or the systems that facilitate communication.
            Internet Services	: Companies that provide a wide variety of products and services primarily online through their web sites.
            Lending and Investments	: Companies that concern themselves with providing debt-based funding to individuals and corporations as well as those that assist individuals and corporations with where/how to invest their money.
            Manufacturing	: Companies that concern themselves with the process of transforming materials or components into finished products that can be sold in the marketplace.
            Media and Entertainment	: A varied collection of companies that share the production, publication, and distribution of media texts.
            Messaging and Telecommunications	: Companies that are involved in the transmission of signs, signals, words, messages, etc.
            Mobile	: Companies that are involved in the manufacturing of mobile phones, including mobile phone handsets and apps.
            Music and Audio	: Companies that earn money by creating new songs and pieces and selling live concerts and shows, audio and video recordings, compositions and sheet music, and the organizations and associations that aid and represent creators.
            Natural Resources	: Companies that are concerned with what people can use which comes from the natural environment.
            Navigation and Mapping	: Companies involved with the processes of monitoring and controlling the movement of a craft or vehicle from one place to another.
            Other	
            Payments	:   Companies in the massive card-processing industry.
            Platforms : Products, services, or technologies that act as a foundation upon which external innovators, organized as innovative business ecosystem, can develop their own complementary products, technologies, or services.
            Privacy and Security : Companies that concern themselves with the ability to protect information about personally identifiable information.
            Professional Services : Companies in the tertiary sector of the economy requiring special training in the arts or sciences.
            Real Estate	: Companies that encompass the many facets of property including development, appraisal, marketing, selling, leasing, and management of commercial, industrial, residential, and agricultural properties.
            Sales and Marketing : 	Companies that include operations and activities involved in promoting and selling goods or services and the process or technique of promoting, selling, and distributing a product or service.
            Science and Engineering : Companies whose main area of focus revolves around the science and/or engineering fields.
            Software : Companies that work on the development, maintenance, and publication of software that are using different business models, mainly either license/maintenance based or cloud based.
            Sports : Companies that are involved in producing, facilitating, promoting, or organizing any activity, experience, or business enterprise focused on sports.
            Sustainability :	Companies concerned with the creation of manufactured products through economically-sound processes that minimize negative environmental impacts while conserving energy and natural resources.
            Transportation : Companies that provide services moving people, goods, or the infrastructure to do so.
            Travel and Tourism	: Companies who are concerned with people traveling for business or pleasure purposes.
            Video	: Companies that primarily concern themselves with producing video content.
            
            Companies to classify:
            {chr(10).join(f'- {name}' for name in company_names)}
            
            Return STRICTLY in this JSON format (NO markdown code blocks, JUST pure JSON):
            {{"company1": "category1", "company2": "category2"}}
            """
            
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_ctx': 4096,
                    'num_gpu': 50  # Use 50 layers on GPU (max for gemma3:12b)
                }
            )
            
            # Clean response by removing markdown code blocks
            raw_response = response['response']
            cleaned_response = raw_response.replace('```json', '').replace('```', '').strip()
            print(f"[DEBUG] Cleaned response: {cleaned_response}")
            
            try:
                results = json.loads(cleaned_response)
                # Validate all companies were classified
                for company in company_names:
                    if company not in results:
                        print(f"[WARNING] Missing classification for {company}")
                        results[company] = "Unknown"
                return results
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON (pos {e.pos}): {cleaned_response}")
                return {company: "Unknown" for company in company_names}
            
        except Exception as e:
            print(f"[ERROR] Batch failed: {str(e)}")
            return {company: "Unknown" for company in company_names}

    def make_gemma_request(self, company_name):
        """Single LLM request with JSON output"""
        try:
            prompt = f"""Classify this company: {company_name}
            
            Categories : Definitions:
            
            Administrative Services :	Companies that are primarily engaged in providing a range of day-to-day office administrative services, such as financial planning; billing and record keeping; and physical distribution and logistics, for others on a contract or fee basis.
            Advertising :	Companies that create and manage the connection between companies, products, and consumers, translating their clients' messages into effective campaigns.
            Agriculture and Farming :	Companies that encompass all aspects of food production.
            Apps :	Companies that build applications.
            Artificial  Intelligence :	Companies that concern themselves with the simulation of human intelligence in machines.
            Biotechnology :	The broad area of biology involving living systems and organisms to develop or make products, or any technological application that uses biological systems, living organisms, or derivatives thereof, to make or modify products or processes for specific use.
            Blockchain & Cryptocurrency :	Companies that relate to decentralized digital currencies or the ledgers on which decentralized financial transactions are recorded.
            Clothing and Apparel :	Companies that design and sell clothing, footwear, and accessories.
            Commerce and Shopping :	Companies involved in the buying and selling of goods by consumers and/or businesses.
            Community and Lifestyle :	Companies that are set up and run by its founders primarily with the aim of sustaining a particular level of income and no more; or to provide a foundation from which to enjoy a particular lifestyle.
            Consumer Electronics :	Companies that produce electronic equipment intended for everyday use, typically in private homes.
            Consumer Goods	: Companies that relate to items purchased by individuals and households rather than by manufacturers and industries.
            Content and Publishing :	Companies that are involved in the creation of and distribution of content.
            Data and Analytics	: Companies that analyze raw data in order to make conclusions about that information.
            Design	: Companies that influence the experience a user has with all of a company's touch points.
            Education	: Companies that provide instruction and training in a wide variety of subjects.
            Energy	: Companies who concern themselves with researching and creating new forms of energy.
            Events	: Companies that have experience in the intricacies and creativity of throwing a memorable and rewarding corporate or social event.
            Financial Services	: Companies that provide a broad range of businesses that manage money.
            Food and Beverage	: Companies involved in processing raw food materials, packaging, and distributing them.
            Gaming	: Companies that involve the development, marketing, and monetization of video games.
            Government and Military	: Companies that are involved with, make things for, and deal with the government and/or military.
            Hardware	: Companies concerned with a physical component of any computer or telecommunications system.
            Health Care	: Companies within the economic system that provide goods and services to treat patients with curative, preventive, rehabilitative, and palliative care.
            Information Technology	: Companies that deal with computing, including hardware, software, telecommunications and generally anything involved in the transmittal of information or the systems that facilitate communication.
            Internet Services	: Companies that provide a wide variety of products and services primarily online through their web sites.
            Lending and Investments	: Companies that concern themselves with providing debt-based funding to individuals and corporations as well as those that assist individuals and corporations with where/how to invest their money.
            Manufacturing	: Companies that concern themselves with the process of transforming materials or components into finished products that can be sold in the marketplace.
            Media and Entertainment	: A varied collection of companies that share the production, publication, and distribution of media texts.
            Messaging and Telecommunications	: Companies that are involved in the transmission of signs, signals, words, messages, etc.
            Mobile	: Companies that are involved in the manufacturing of mobile phones, including mobile phone handsets and apps.
            Music and Audio	: Companies that earn money by creating new songs and pieces and selling live concerts and shows, audio and video recordings, compositions and sheet music, and the organizations and associations that aid and represent creators.
            Natural Resources	: Companies that are concerned with what people can use which comes from the natural environment.
            Navigation and Mapping	: Companies involved with the processes of monitoring and controlling the movement of a craft or vehicle from one place to another.
            Other	
            Payments	:   Companies in the massive card-processing industry.
            Platforms : Products, services, or technologies that act as a foundation upon which external innovators, organized as innovative business ecosystem, can develop their own complementary products, technologies, or services.
            Privacy and Security : Companies that concern themselves with the ability to protect information about personally identifiable information.
            Professional Services : Companies in the tertiary sector of the economy requiring special training in the arts or sciences.
            Real Estate	: Companies that encompass the many facets of property including development, appraisal, marketing, selling, leasing, and management of commercial, industrial, residential, and agricultural properties.
            Sales and Marketing : 	Companies that include operations and activities involved in promoting and selling goods or services and the process or technique of promoting, selling, and distributing a product or service.
            Science and Engineering : Companies whose main area of focus revolves around the science and/or engineering fields.
            Software : Companies that work on the development, maintenance, and publication of software that are using different business models, mainly either license/maintenance based or cloud based.
            Sports : Companies that are involved in producing, facilitating, promoting, or organizing any activity, experience, or business enterprise focused on sports.
            Sustainability :	Companies concerned with the creation of manufactured products through economically-sound processes that minimize negative environmental impacts while conserving energy and natural resources.
            Transportation : Companies that provide services moving people, goods, or the infrastructure to do so.
            Travel and Tourism	: Companies who are concerned with people traveling for business or pleasure purposes.
            Video	: Companies that primarily concern themselves with producing video content.
            
  
            Companies to classify:
            {chr(10).join(f'- {name}' for name in company_names)}
            
            Return JSON format: {{"company": "category"}}            """
            
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_gpu': 50  # Use 50 layers on GPU (max for gemma3:12b)
                }
            )
            
            # Debug output
            debug_info = {
                'company': company_name,
                'model': response.get('model'),
                'response': response.get('response'),
                'duration': response.get('total_duration')
            }
            print(f"[DEBUG] {debug_info}")
            
            # Parse as JSON
            result = json.loads(response['response'])
            category = result['category']
            
            if not self._verify_category(category):
                print(f"[WARNING] Invalid category for {company_name}: {category}")
                category = "Unknown"
                
            return {"company": company_name, "category": category}
            
        except Exception as e:
            print(f"[ERROR] Request failed: {str(e)}")
            return {"company": company_name, "category": "Unknown"}

    def _adjust_batch_size(self, success):
        """Adjust batch size within safe limits"""
        if success and self.current_batch_size < self.max_batch_size:
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 2)
        elif not success:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 5)

    def categorize_companies(self, company_names):
        """Process companies with accurate progress tracking"""
        existing_results = self._load_progress().get('results', {})
        raw_responses = {}
        
        # Filter unprocessed companies
        remaining_companies = [
            name for name in company_names 
            if name not in existing_results
        ]
        
        total_processed = len(existing_results)
        total_companies = total_processed + len(remaining_companies)
        
        print(f"- Total companies: {total_companies}")
        print(f"- Already processed: {total_processed}")
        print(f"- Remaining to process: {len(remaining_companies)}")
        print(f"- Using batch size: {self.current_batch_size} (fixed)")
        print(f"- Output file: {self.output_file}\n")
        
        # Track companies processed, not batches
        processed = 0
        with tqdm(total=total_companies, initial=total_processed) as pbar:
            for i in range(0, len(remaining_companies), self.current_batch_size):
                batch = remaining_companies[i:i + self.current_batch_size]
                batch_results = self.make_batch_request(batch)
                
                # Update progress
                processed += len(batch)
                pbar.update(len(batch))
                
                # Save results
                existing_results.update(batch_results)
                raw_responses.update({company: batch_results[company] for company in batch})
                
                # Save progress
                self._save_progress({'processed': list(existing_results.keys()), 'results': existing_results}, raw_responses)
                
        return existing_results, raw_responses

    def _load_progress(self):
        """Load progress from file"""
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'processed': [], 'results': {}}

    def _save_progress(self, processed_data, raw_data=None):
        """Save progress to file"""
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(processed_data, f)
            # Save both output files
            with open(self.output_file, 'w') as f:
                json.dump(processed_data, f)
            if raw_data is not None:
                with open(self.raw_file, 'w') as f:
                    json.dump(raw_data, f)
        except Exception as e:
            print(f"Warning: Failed to save progress - {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description='Company categorization script')
    parser.add_argument('--input', default='company_names.csv', help='Input CSV file path')
    parser.add_argument('--output', default='company_categories_gemma.json', 
                       help='Output JSON file for validated categories')
    parser.add_argument('--raw-output', default='company_categories_raw.json',
                       help='Output JSON file for raw model responses')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of companies to process per batch (default: 10)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    categorizer = CompanyCategorizerGemma(
        output_file=args.output,
        raw_file=args.raw_output,
        batch_size=args.batch_size
    )
    
    try:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, 'r') as f:
                existing_results = json.load(f)
            print(f"Resuming with {len(existing_results)} pre-processed companies")
    except Exception:
        existing_results = {}
    
    # Read companies from specified input file
    df = pd.read_csv(args.input, header=None)
    all_companies = df[0].str.strip().dropna().tolist()
    
    # Process with specified batch size
    categories, raw_responses = categorizer.categorize_companies(all_companies)

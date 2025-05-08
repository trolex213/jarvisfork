import torch
print(f"ROCm version: {torch.version.hip}")  # Verify ROCm is active
import json
import re
import pdfplumber
import os
import ollama
import time
from datetime import datetime
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

# Configuration
OLLAMA_MODEL = "gemma3:12b"
MAX_RETRIES = 3

# Define output schema matching LinkedIn profile format
response_schemas = [
    ResponseSchema(name="name", description="Full name"),
    ResponseSchema(name="city", description="City, State, Country"),
    ResponseSchema(name="country_code", description="Country code"),
    ResponseSchema(name="current_company_name", description="Current company name"),
    ResponseSchema(name="position", description="Current position"),
    ResponseSchema(name="professional_keywords", description="Professional Key skills"),
    ResponseSchema(name="personal_keywords", description="Personal Key skills"),
    ResponseSchema(name="is_student", description="True if currently a student"),
    ResponseSchema(name="generated_summary", description="Summary of resume including based on professional and personal experience"),
    ResponseSchema(name="experience_years", description="Total years of experience"),
    ResponseSchema(name="honors_and_awards", description="Honors and Awards"),
    ResponseSchema(name="projects", description="personal or professional projects"),
    ResponseSchema(name="education", description="List of education entries", type="array"),
    ResponseSchema(name="experience", description="List of work experiences", type="array"),
    ResponseSchema(name="organizations", description="List of organizations / clubs", type="array"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

def extract_text_from_pdf(pdf_path):
    """Extracts and cleans text from PDF resume"""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def build_resume_prompt(resume_text):
    """Constructs parsing prompt with LangChain formatting"""
    template="""Extract resume details from this text:
{resume_text}

{format_instructions}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["resume_text"],
        partial_variables={"format_instructions": format_instructions}
    )
    return prompt

def save_to_json(data, pdf_path):
    """Save parsed data to JSON file"""
    output_dir = os.path.join(os.path.dirname(pdf_path), 'parsed')
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'{base_name}_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

def parse_resume_with_retry(pdf_path):
    """Main parsing function with retry logic"""
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("No text extracted from PDF")
            
        print(f"\nExtracted text (first 200 chars):\n{text[:200]}...\n")
        prompt = build_resume_prompt(text)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt.format(resume_text=text),
                    options={
                        'num_gpu': -1,
                        'num_ctx': 4096,
                        'temperature': 0.1,
                        'seed': 42
                    }
                )
                
                # Parse with LangChain
                parsed = output_parser.parse(response['response'])
                output_path = save_to_json(parsed, pdf_path)
                print(f"\nSuccess! Saved to: {output_path}")
                return parsed
                
            except Exception as e:
                print(f"\nAttempt {attempt + 1} failed: {str(e)}")
                print(f"Raw response: {response['response']}")
                if attempt < MAX_RETRIES - 1:
                    print("Retrying...\n")
                    time.sleep(1)
            
        print("\nAll attempts failed.")
        return None
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

def generate_connection_points(resume_data):
    """Generate interesting connection points from parsed resume data"""
    if not resume_data:
        return None
        
    prompt = """give me 10 interesting things in this person's resume that could have a unique connection with other people. 
list in order of likelihood to connect (and how strong the connection will be). 
For the first two, always put university and high school.

Resume Data:
{resume_data}

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "connection_points": [
    {{
      "item": "description of item",
      "reason": "why it's interesting",
      "connection_strength": 0-10,
      "connection_potential": "description"
    }}
  ]
}}""".format(resume_data=json.dumps(resume_data, indent=2))
    
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                'num_gpu': -1,
                'num_ctx': 4096,
                'temperature': 0.7,
                'seed': 42
            }
        )
        
        # Clean and parse response
        raw_response = response['response'].strip()
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:-3].strip()
        
        return json.loads(raw_response)
        
    except Exception as e:
        print(f"\nError generating connections: {str(e)}")
        print(f"Raw response: {response['response']}")
        return None

if __name__ == "__main__":
    pdf_path = "resumes/Alicia_Wang_Resume.pdf"
    resume_data = parse_resume_with_retry(pdf_path)
    
    if resume_data:
        print("\nGenerating connection points...")
        connections = generate_connection_points(resume_data)
        if connections:
            print("\nPotential connection points:")
            print(json.dumps(connections, indent=2))
            
            # Save connections to file
            connections_path = save_to_json(
                {"resume_data": resume_data, "connections": connections},
                pdf_path.replace(".pdf", "_connections.json")
            )
            print(f"\nSaved connections to: {connections_path}")

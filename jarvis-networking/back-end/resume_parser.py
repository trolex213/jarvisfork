import pdfplumber
from docx import Document
import openai

def extract_resume_text(file_path):
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    raise ValueError("Unsupported file")

def extract_json(text):
    openai.api_key = "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"

    prompt = f"""
You are a resume parser. Extract the following sections from the resume provided by the user and return JSON:
- name
- contact (email, phone, location)
- education (block of text or structured per institution)
- experience (block of text or structured per job)
- skills (block of text or grouped)
- certifications (if any)
- languages
- interests

Example output format:
{{
  "name": "John Doe",
  "contact": {{
    "email": "blahblah@gmail.com",
    "phone": "(999)999-9999",
    "location": "Boston, MA"
  }},
  "education": {{
    "Some University": {{
      "degree": "Master of Professional Studies in Management",
      "date": "August 2022 - May 2023",
      "GPA": "3.85/4.00",
      "coursework": "SQL | Advanced Corporate Finance | Power BI | Financial Modeling | Tableau | Investment"
    }},
    "New University": {{
      "degree": "Bachelor of Science, Economics, Theory Concentration",
      "date": "August 2019 - May 2022",
      "GPA": "3.95/4.00",
      "coursework": "Econometrics | Microeconomics | Accounting | Macroeconomics | Marketing | Finance | Python",
      "honors": "Macromanagement"
    }}
  }},
  "experience": [
    {{
      "company": "AMERICAN EAGLE",
      "location": "Woburn, MA",
      "title": "Full-time Consultant",
      "date": "July 2023 - Now",
      "responsibilities": "- Collaborated with the store merchandiser creating displays to attract clientele
- Use my trend awareness to assist customers in their shopping experience
- Thoroughly scan every piece of merchandise for inventory control
- Process shipment to increase my product knowledge"
    }},
    {{
      "company": "Some Investment Management Co., Ltd",
      "location": "Shenzhen, China",
      "title": "Investment Management Intern, Private Equity",
      "date": "January 2021 - April 2021",
      "responsibilities": "- Sell retail and memberships to meet company sales goals
- Build organizational skills by single handedly running all operating procedures
- Communicate with clients to fulfill their wants and needs
- Attend promotional events to market our services
- Handle cash and deposits during opening and closing
- Received employee of the month award twice"
    }},
    {{
      "company": "Some Bank",
      "location": "Beijing, China",
      "title": "Intern",
      "date": "September 2020 - November 2020",
      "responsibilities": "- Applied my leadership skills by assisting in the training of coworkers
- Set up mannequins and displays in order to entice future customers
- Provided superior customer service by helping with consumer decisions
- Took seasonal inventory"
    }}
  ],
  "skills": [
    {{
      "Language Skills": "Mandarin (native); English (fluent)",
      "Computer Skills": "SQL (intermediate); Power BI (advanced); Tableau (advanced); Microsoft Excel (advanced); Python (intermediate)"
    }}
  ],
  "interests": ["Stock market", "Hiking", "Bowling", "Archery", "Tennis"],
  "certifications": ["CFA Level 3 passed (all 1st attempt)", "FRM Part 2 passed (all 1st attempt)"]
}}
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

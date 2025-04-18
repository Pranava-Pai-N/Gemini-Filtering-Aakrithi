from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import requests
import os
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-001:generateContent?key={GEMINI_API_KEY}"

class RoutineItem(BaseModel):
    time: str
    content: str

class InputData(BaseModel):
    title: str
    description: str
    media: List[str]
    routines: List[RoutineItem] = []

CUSTOM_PROMPT = """
You are an AI assistant specializing in content classification.
Your task is to analyze and categorize a given post into one or more predefined filters.

### Step-by-step Instructions:

1. **Verify Post Structure**  
   Ensure the post includes:
   - A `title`
   - A `description`
   - At least one media item (either a base64-encoded string or a valid URL)

2. **Check for Diseases**  
   Identify whether the post explicitly mentions any disease (e.g., viral fever, acidity, stress, cancer, diabetes, etc.).  
   → If found, extract and return the disease name in lowercase (e.g., `"cancer"`).  
   → If the disease name contains multiple words, separate them with a single space.

3. **Check for Ayurvedic Medicines**  
   Look for mentions of specific Ayurvedic medicines (e.g., Ashwagandha, Brahmi, Neem, Triphala, etc.).  
   → Return each medicine name in lowercase as a separate item in the result.

4. **Apply Custom Filters (only if relevant)**  
   Based on the content, classify the post using only the following filter keys:
const categories = [
    { value: "herbs", give to frontend: "Herbs & Remedies" },
    { id: "routines", label: "Daily Routines" },
    { id: "wellnessTips", label: "Wellness Tips" },
    { id: "diet", label: "Diet & Nutrition" },
    { id: "yoga", label: "Yoga & Pranayama" },
    { id: "detox", label: "Detox & Cleansing" },
    { id: "seasonal", label: "Seasonal Care" }
];

### Output Format (STRICT)
- Return a **flat JSON array** of strings.
- Example valid output:
["cancer", "ashwagandha", "herbs", "routines"]

- Do NOT include:
  - Markdown formatting
  - Explanations
  - Headers or bullet points
  - Code blocks (like ```json)
"""

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Filter Generator!"}


@app.post("/generate_filters")
async def generate_filters(data: InputData):
    formatted_routines = "\n".join([f"- {item.time}: {item.content}" for item in data.routines]) or "None"
    user_message = f"""
Title: {data.title}
Description: {data.description}
Routines:\n{formatted_routines}
Media Count: {len(data.media)}

{CUSTOM_PROMPT}
"""
    payload = {
        "contents": [
            {
                "parts": [{"text": user_message}]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    result = response.json()

    try:
        if "candidates" not in result:
            raise ValueError(f"Gemini response missing 'candidates': {result}")

        raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        cleaned_text = raw_text.strip("`json").strip("`").strip()
        filters = json.loads(cleaned_text)

        if not isinstance(filters, list):
            raise ValueError("Response is not a valid JSON list")

        return filters

    except Exception as e:
        return {
            "error": "Failed to parse filters",
            "details": str(e),
            "raw_response": result
        }

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict
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
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

class InputData(BaseModel):
    title: str
    description: str
    media: List[str]
    routines: Union[str, Dict[str, str]] = ""

CUSTOM_PROMPT = """
You are an AI assistant specializing in content classification.
Your task is to categorize a given post into one or more *predefined filters*.

First, check if the post includes title, description, and media (base64 or URL). If the post mentions a disease(eg:stress,viral fever,headache etc) or Ayurvedic medicine, return the name of the disease or Ayurvedic medicine first if more than one found return all the names. Then, if applicable, check for the following predefined filters and sub-filters:

const categories = [
    { value: 'herbs', give to frontend: 'Herbs & Remedies' },
    { id: 'routines', label: 'Daily Routines' },
    { id: 'wellnessTips', label: 'Wellness Tips' },
    { id: 'diet', label: 'Diet & Nutrition' },
    { id: 'yoga', label: 'Yoga & Pranayama' },
    { id: 'detox', label: 'Detox & Cleansing' },
    { id: 'seasonal', label: 'Seasonal Care' }
];

*Give to frontend* means: return only the filter key/label in valid JSON format.

If none of the filters like disease, medicine, or sub-filters match, return the closest matching one from the above list.

### Task Requirements
1. If a disease or Ayurvedic medicine is mentioned, return its name first.
2. After identifying any disease or Ayurvedic medicine, proceed to check the filters.
3. Only use filters from the list above. Do NOT create new ones.
4. Output must be strictly valid JSON — no extra text or markdown.
5. Example valid response: ["herbs", "diet"]

Give all output in valid JSON format without any additional text or explanation in lowercase.
"""

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Filter Generator!"}

@app.post("/generate_filters")
async def generate_filters(data: InputData):
    user_message = f"""
Title: {data.title}
Description: {data.description}
Routines: {data.routines if data.routines else 'None'}
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

    # Call Gemini API
    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    result = response.json()

    try:
        if "candidates" not in result:
            raise ValueError(f"Gemini response missing 'candidates': {result}")

        raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Clean the raw text to remove unwanted code block formatting
        # Remove the '```json' and '```' markers
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Parse the cleaned text into JSON
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

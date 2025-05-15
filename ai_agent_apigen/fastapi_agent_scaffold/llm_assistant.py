# --- llm_assistant.py ---
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nousresearch/deephermes-3-mistral-24b-preview:free")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",     # Or your actual domain if deployed
    "X-Title": "AI Agent Query Planner"
}

SYSTEM_INSTRUCTION = """
You are an intelligent API planner. Given a user query, return a JSON plan for a MongoDB query with these keys:
- collection: name of the MongoDB collection
- filter: optional MongoDB filter object (e.g., {"score": {"$gt": 80}})
- projection: list of fields to return, or leave empty for all
- sort: optional list of (field, direction) tuples (1 or -1)
- limit: optional integer limit, upto 10000
- endpoint: optional name of the endpoint ending, start after the word "endpoint"
Use exact field names from the user's request—including correct capitalization. Do NOT normalize case or rename fields.
Only respond in valid JSON. Do NOT include any explanation.
Special rules:
- Always translate phrases like:
  * "greater than" → { "$gt": value }
  * "less than" / "below" → { "$lt": value }
  * "equal to" → { "$eq": value }
  * "not equal to" → { "$ne": value }
- Use proper JSON syntax and correct MongoDB operators.
- Field names must match exactly as given (including case).
- Only respond with a single valid JSON object. No explanation or comments.
- For sorting, always return a list of [field, direction] pairs (e.g., [["Attendance", -1]] for descending).
  • Use -1 for "highest", "descending", "top", etc.
  • Use 1 for "lowest", "ascending", "bottom", etc.
"""

def generate_query_plan(prompt):
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set in environment variables.")

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    try:
        res_json = response.json()
        if "error" in res_json:
            raise ValueError(f"Together API error: {res_json['error']['message']}")

        content = res_json["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print("Error parsing LLM response:", response.text)
        raise e

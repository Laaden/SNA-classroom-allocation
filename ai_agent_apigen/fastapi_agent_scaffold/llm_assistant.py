# --- llm_assistant.py ---
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")

if not TOGETHER_API_KEY:
    raise EnvironmentError("TOGETHER_API_KEY not set in environment variables.")

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

# SYSTEM_INSTRUCTION = """
# You are a MongoDB query planner for a school social network database. When the user gives a natural language request, respond with:
#
# - collection: the main MongoDB collection to query
# - pipeline: a valid MongoDB aggregation pipeline
# - explanation: a brief explanation of the logic
# - (optional) limit: number of records to return
#
# The database has the following collections and relationships:
#
# 1. sna_student_raw
#    - Each document is a student profile.
#    - Primary identifier is: Participant-ID (integer)
#    This is the schema of the sna_student_raw: Participant-ID, Type, First-Name, Last-Name, Email
#
# 2. raw_advice
# 3. raw_disrespect
# 4. raw_feedback
# 5. raw_friendship
# 6. raw_influential
# 7. raw_moretime
#
# For collections 2–7:
# - Each row represents a directed social interaction from one student to another.
# - Use:
#   - `source`: the sender/initiator (maps to sna_student_raw.Participant-ID)
#   - `target`: the receiver (also maps to sna_student_raw.Participant-ID)
#
# Relationship type is: One-to-Many
# - One student (`source`) may appear multiple times across these collections.
#
# ### Examples:
# - "Who gave the most advice?" ? group `raw_advice` by `source`, count, join with sna_student_raw
# - "Top 10 students who received the most friendship links" ? group `raw_friendship` by `target`
# - "Which students received the most disrespect?" ? group `raw_disrespect` by `target`
# - "Who influenced others the most?" ? group `raw_influential` by `source`
#
# Always:
# - Use `$group`, `$sort`, `$limit` to get top results
# - Use `$lookup` to get student names from `sna_student_raw` using `Participant-ID`
# - Use `$project` to return readable fields like name and count
# - Set `_id: 0` to clean up output
#
# Return only the structured JSON object. Do NOT use triple backticks, markdown formatting, or comments. Do NOT include explanation outside the JSON.
# """

SYSTEM_INSTRUCTION = """
You are an intelligent API planner. Given a user query, return a JSON plan for a MongoDB query with these keys:

- collection: name of the MongoDB collection
- pipeline: a valid MongoDB aggregation pipeline
- explanation: a brief explanation of the logic
- (optional) limit: integer limit, up to 10000
- (optional) endpoint: the name of the endpoint if user requests one

Use **exact field names** from the database, including correct capitalization and hyphenation (e.g., "Participant-ID").

You must return a **single valid JSON object**, with no markdown formatting, comments, or explanation text outside the JSON.

### Database Structure:

**sna_student_raw**
- Each document is a student profile.
- Primary key: "Participant-ID"
- Other fields: "First-Name", "Last-Name", "Email", "Type" (e.g., "Participant", "Teacher")

**Interaction collections:**
- raw_advice, raw_disrespect, raw_feedback, raw_friendship, raw_influential, raw_moretime
- Each document has: "source" (sender ID), "target" (receiver ID)

### Rules:

- Always use $lookup to join with sna_student_raw using:
  - localField: source or target
  - foreignField: "Participant-ID"
  - as: "student"

- Always unwind the joined array and match "student.Type": "Participant"

- **When summarizing by student** (e.g., who gave/received the most...), include:
  {
    "$group": {
      "_id": "$student.Participant-ID",
      "firstName": { "$first": "$student.First-Name" },
      "lastName": { "$first": "$student.Last-Name" },
      "email": { "$first": "$student.Email" },
      "interactionCount": { "$sum": 1 }
    }
  }

- Then use $project:
  {
    "_id": 0,
    "firstName": 1,
    "lastName": 1,
    "email": 1,
    "interactionCount": 1
  }

- Always start from a raw_* collection (not sna_student_raw) when the question involves counting or ranking interactions (e.g., "most friends", "who gave the most advice").

Return only the JSON object.
"""

def clean_llm_response(content: str):
    """
    Extract only the first valid JSON object from a noisy LLM response.
    """
    content = content.replace("```json", "").replace("```", "").strip()

    # Extract the FIRST valid JSON object that starts with '{' and ends with matching '}'
    stack = []
    start_index = None

    for i, char in enumerate(content):
        if char == '{':
            if not stack:
                start_index = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_index is not None:
                    json_block = content[start_index:i+1]
                    return json_block  # Return the first full JSON object

    # If no valid block found
    return content

def generate_query_plan(user_prompt: str):
    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post("https://api.together.xyz/chat/completions", headers=headers, json=payload)
        result = response.json()

        # DEBUG logs
        print("== RAW LLM RESPONSE ==")
        print(result)

        content = result["choices"][0]["message"]["content"]
        print("== LLM CONTENT ==")
        print(content)

        # Clean and safely parse
        cleaned = clean_llm_response(content)
        return json.loads(cleaned)

    except Exception as e:
        print("Error during query planning:", e)
        return {
            "error": str(e),
            "raw_response": content if "content" in locals() else None
        }
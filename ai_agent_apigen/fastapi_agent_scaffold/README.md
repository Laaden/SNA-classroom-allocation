# 🧠 AI-Powered FastAPI Endpoint Generator

## 📘 Project Overview

This project enables users to create FastAPI-based MongoDB endpoints using **natural language prompts**. By integrating a language model (LLM) with dynamic FastAPI routing and MongoDB queries, users can describe the desired data access behavior in plain English, and the system will auto-generate and expose an API endpoint accordingly.

Typical user prompt example:
> *"Create an endpoint for `students` to return all records where `score` is greater than 85, show `name` and `ID`, and name the endpoint `top_students`."*

---

## 📂 Project Structure

```
fastapi_agent_project/
│
├── main.py                  # FastAPI application: handles GUI + endpoint injection
├── llm_assistant.py         # LLM wrapper to generate query plan from user prompt
├── test.py                  # Manual test script to insert mock MongoDB data
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (e.g., MongoDB URI, API key)
│
├── templates/               # HTML frontend
│   ├── index.html           # Prompt input form
│   └── result.html          # Endpoint generation result page
│
└── static/                  # Static assets (optional, currently unused)
```

---

## ⚙️ Configuration Setup

You’ll need to configure the following in your `.env` file:

```ini
MONGO_URI=mongodb://<username>:<password>@<host>:27017/?authSource=admin
DB_NAME=sna_database
TOGETHER_API_KEY=your_api_key_here
```

> 🔐 Keep this file secure. Do **not** upload it to public repositories.

**Dependencies:**
Install with pip:
```bash
pip install -r requirements.txt
```

Main packages:
- `fastapi`, `uvicorn`: web server
- `pymongo`: MongoDB interaction
- `requests`: API call to LLM
- `jinja2`: HTML templating
- `python-dotenv`: environment loader

---

## 🚀 Deployment Instructions (Detailed)

### 1. Clone the Project
```bash
git clone https://your-repo-url
cd fastapi_agent_project
```

### 2. Set Up Virtual Environment (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Configure `.env`
Edit the `.env` file with:
- your MongoDB URI
- database name
- Together.ai API key

### 5. Run the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```

Access the app at: [http://localhost:8000](http://localhost:8000)

---

## 🧑‍💻 Using the GUI

### Home Page

When you visit the app (`http://localhost:8000`), you'll see a simple prompt form.

### Step-by-Step Instructions

1. **Enter a Prompt**  
   Example:
   > Create an endpoint for `sna_student_raw` to return the top 20 records with the highest `Attendance`, including `First-Name`, `Last-Name`, and `ID`. Name the endpoint `top_attendees`.

2. **Click "Generate Endpoint"**  
   The system will:
   - Send the prompt to the LLM API
   - Parse the response into MongoDB query format
   - Dynamically create a new FastAPI route
   - Display the access URL

3. **Use the URL to Fetch Data**  
   After generation, the result page will show something like:
   > `Access your new endpoint at: /sna_student_raw/top_attendees`

   You can click that URL to view the data immediately.

4. **Repeat**  
   You can generate multiple endpoints — each one is independent and immediately usable.
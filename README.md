# 📄 AI Resume Screening Assistant (LangChain & RAG)

An intelligent, lightweight resume screening system designed for recruiters and hiring managers. It leverages **LangChain**, **FAISS Vector Store**, **Google Gemini (`gemini-3.5-flash-lite`)**, and **Streamlit** to evaluate candidate PDF resumes against a specific Job Description (JD) using Retrieval-Augmented Generation (RAG).

---

## 🚀 Live Demo & Deployment

Explore the live application deployed on Streamlit Community Cloud:
👉 **[resume-screening-assistant-langchain.streamlit.app](https://resume-screening-assistant-langchain.streamlit.app)**

---

## 🏗️ Architecture & Pipeline Flow

1. **Document Loading:** Ingests PDF candidate resumes using `PyPDFLoader`.
2. **Text Splitting:** Breaks down resume text into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embeddings & Vector Store:** Converts chunks into vector embeddings via Google Gemini (`models/gemini-embedding-001`) and stores them locally in a `FAISS` vector database.
4. **Retrieval & Generation:** Retrieves contextually relevant resume excerpts matching the Job Description and queries `gemini-3.5-flash-lite` with strict prompt constraints to prevent hallucinations.
5. **Structured Output:** Parses the LLM response into a clean, validated JSON schema (Candidate Name, Match Score, Matching & Missing Skills, Summary, Strengths, Weaknesses, and Hiring Recommendation) using Pydantic.

---

## 📂 Project Structure

```text
resume-screening-assistant/
├── app.py              # Main interactive Streamlit web application
├── project.ipynb       # Jupyter Notebook demonstrating end-to-end RAG evaluations
├── requirements.txt    # Python package dependencies
├── .env                # Local environment variables configuration
└── README.md           # Project documentation
```

---

⚙️ System Prerequisites

- Python 3.10 or higher installed on your machine.
- A Google Gemini API Key. (You can obtain one for free in seconds from [Google AI Studio](https://aistudio.google.com/app/apikey)).

🛠️ Setup & Installation Instructions

- Choose either Method A (uv recommended for speed) or Method B (traditional pip).

Method A: Using uv (Extremely Fast)

1. Install uv if it is not already installed on your system:

```bash
pip install uv
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Sync and install all required dependencies:

```bash
uv pip install -r requirements.txt
```

Method B: Using Standard pip

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

▶️ Running the Project
Step 1: Configure Environment Variables
Create a file named `.env` in the root directory of your project folder and add your Gemini API key:

```env
GOOGLE_API_KEY="your_actual_gemini_api_key_here"
```

*(Note: The Streamlit app also provides a secure input box in the UI if you prefer entering your API key interactively at runtime).*

Step 2: Run the Jupyter Notebook
To run programmatic evaluations and automatically generate sample test resumes (Resume_A.pdf, Resume_B.pdf, Resume_C.pdf):

```bash
jupyter notebook project.ipynb
```

Step 3: Launch the Streamlit Web Application
To run the interactive user interface locally:

```bash
streamlit run app.py
```

Open your browser and navigate to the local URL displayed in your terminal (usually http://localhost:8501).

💡 Troubleshooting

- Missing API Key Error: Ensure your .env file is placed in the exact root folder of the project, or paste your key directly into the Streamlit input field.
- FAISS / NumPy compatibility errors: Ensure you are using the exact package versions specified in requirements.txt (faiss-cpu).

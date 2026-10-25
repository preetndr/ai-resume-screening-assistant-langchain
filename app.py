import tempfile
from typing import List
import streamlit as st
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

st.set_page_config(page_title="AI Resume Screening Assistant", layout="wide")
st.title("📄 AI Resume Screening Assistant (LangChain + RAG)")

# ---------------------------------------------------------
# User Google API Key Section + Direct Link to Google AI Studio
# ---------------------------------------------------------
st.subheader("🔑 Google Gemini API Key")
col_key, col_link = st.columns([2, 1])
with col_key:
    user_api_key = st.text_input(
        "Enter your Google Gemini API Key to use the app:",
        type="password",
        placeholder="Paste your Gemini API key here..."
    )
with col_link:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "👉 **[Get a free Gemini API Key from Google AI Studio](https://aistudio.google.com/app/apikey)**"
    )

if not user_api_key:
    st.info("Please enter your Google Gemini API Key above to start. Click the link next to the box to get one from Google AI Studio.")
    st.stop()

# Initialize Gemini 3.5 Flash Lite and Embeddings with the user's API key
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=user_api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=user_api_key)

# ---------------------------------------------------------
# Structured Output Schema & RAG Pipeline
# ---------------------------------------------------------
class ResumeEvaluation(BaseModel):
    candidate_name: str = Field(description="Candidate name from the resume")
    match_score: int = Field(description="Match score from 0 to 100")
    matching_skills: List[str] = Field(description="Matching skills found in the resume")
    missing_skills: List[str] = Field(description="Required skills from JD missing in the resume")
    candidate_summary: str = Field(description="Short summary of the candidate")
    strengths: List[str] = Field(description="Key strengths")
    weaknesses: List[str] = Field(description="Key weaknesses")
    hiring_recommendation: str = Field(description="Hire / Reject recommendation with justification")

parser = JsonOutputParser(pydantic_object=ResumeEvaluation)

rag_prompt = ChatPromptTemplate.from_template(
    "You are an AI Resume Screening Assistant. Answer ONLY from the uploaded resume context below.\n\n"
    "Job Description:\n{jd}\n\n"
    "Resume Context:\n{context}\n\n"
    "{format_instructions}"
)

def evaluate_resume(pdf_path: str, jd: str) -> dict:
    # 1. PDF Document Loader
    docs = PyPDFLoader(pdf_path).load()
    # 2. Text Splitter
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    # 3. Vector Database (FAISS) & Retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(jd)
    context = "\n\n".join([d.page_content for d in retrieved_docs])
    # 4. LLM Chain + Output Parser
    chain = rag_prompt | llm | parser
    return chain.invoke({
        "jd": jd,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
jd_input = st.text_area(
    "Enter Job Description (JD):",
    value="Role: Data Scientist. Required Skills: Python, SQL, Machine Learning, Deep Learning, PyTorch, LangChain, RAG, FAISS.",
    height=120
)

uploaded_pdfs = st.file_uploader("Upload One or More Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Evaluate Resumes") and uploaded_pdfs:
    results = []
    for pdf_file in uploaded_pdfs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Evaluating {pdf_file.name}..."):
            res = evaluate_resume(tmp_path, jd_input)
            res["file_name"] = pdf_file.name
            results.append(res)

        st.subheader(f"📌 {pdf_file.name} — Match Score: {res['match_score']}/100")
        st.write(f"**Candidate Name:** {res['candidate_name']}")
        st.write(f"**Candidate Summary:** {res['candidate_summary']}")
        st.write(f"**Matching Skills:** {', '.join(res['matching_skills'])}")
        st.write(f"**Missing Skills:** {', '.join(res['missing_skills'])}")
        st.write(f"**Strengths:** {', '.join(res['strengths'])}")
        st.write(f"**Weaknesses:** {', '.join(res['weaknesses'])}")
        st.write(f"**Hiring Recommendation:** {res['hiring_recommendation']}")
        st.divider()

    if len(results) > 1:
        best = max(results, key=lambda x: x["match_score"])
        st.success(
            f"🏆 Best Candidate Recommendation: {best['candidate_name']} ({best['file_name']}) "
            f"with Match Score {best['match_score']}/100 — {best['hiring_recommendation']}"
        )
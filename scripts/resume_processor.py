import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import json

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

def calculate_similarity_with_all_jds(jd_list, resume_text):
    """
    Compare a resume with all JDs and return the JD with the highest similarity.
    """
    all = []
    for jd in jd_list:
        jd_text = jd["text"]
        similarity = calculate_similarity(jd_text, resume_text)
        match = {
            "jd_name": jd["name"],
            "similarity_score": similarity["overall_similarity"],
            "details": similarity,
        }
        all.append(match)

    return all

# --- Step 1: Read PDF and Extract Text ---
def read_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# --- Step 2: Preprocessing ---
def preprocess_text(text):
    # Remove special characters and convert to lowercase
    return re.sub(r'[^\w\s]', '', text).lower()

# --- Step 3: Calculate Similarity ---
def calculate_similarity(job_description_text, resume_text):
    """
    Compare the job description and resume using text embeddings and experience.
    """
    job_text = preprocess_text(job_description_text)
    resume_text = preprocess_text(resume_text)

    # Convert text into embeddings
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    # Compute cosine similarity for text
    text_similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()

    # Calculate experience similarity
    job_experience = extract_years_of_experience(job_description_text)
    resume_experience = extract_years_of_experience(resume_text)
    experience_similarity = calculate_experience_similarity(job_experience, resume_experience)

    # Weighted overall similarity
    overall_similarity = 0.7 * text_similarity + 0.3 * experience_similarity

    return {
        "text_similarity": round(text_similarity, 2),
        "experience_similarity": round(experience_similarity, 2),
        "overall_similarity": round(overall_similarity, 2),
    }

def extract_years_of_experience(text):
    """
    Extract the years of experience from the text.
    """
    match = re.search(r'(\d+)\+?\s*years?', text.lower())
    return int(match.group(1)) if match else 0

def calculate_experience_similarity(job_experience, resume_experience):
    """
    Calculate similarity based on experience between job description and resume.
    """
    if job_experience == 0:  # No experience requirement
        return 1
    difference = max(0, job_experience - resume_experience)
    return max(0, 1 - (difference / job_experience))


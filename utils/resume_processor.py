import pdfplumber
from sentence_transformers import SentenceTransformer, util
from .resume_analysis import extract_resume_fields, calculate_similarity

# Initialize NLP model
model = SentenceTransformer("all-mpnet-base-v2")

def process_resume(resume_path, job_id, mongo):
    # Extract text from resume PDF
    with pdfplumber.open(resume_path) as pdf:
        resume_text = ''.join(page.extract_text() for page in pdf.pages)

    # Retrieve job description from the database
    job_description = mongo.db.job_descriptions.find_one({"job_id": job_id})
    if not job_description:
        return {"error": f"Job Description with ID {job_id} not found."}

    job_text = job_description.get("description", "")

    # Extract fields from the resume
    extracted_fields = extract_resume_fields(resume_text)

    # Calculate similarity
    similarity_scores = calculate_similarity(model, job_text, resume_text)

    return {
        "filename": resume_path.split("/")[-1],
        "similarity": similarity_scores,
        "extracted_fields": extracted_fields
    }

def process_resume_batch(resume_paths, job_id, mongo):
    results = []
    for path in resume_paths:
        results.append(process_resume(path, job_id, mongo))
    return results

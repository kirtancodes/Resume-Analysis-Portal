import re

def extract_resume_fields(resume_text):
    # Example of extracting fields like phone, email, etc.
    phone_pattern = r"\+?[0-9][0-9.\-()\s]{8,}[0-9]"
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    phone = re.findall(phone_pattern, resume_text)
    email = re.findall(email_pattern, resume_text)

    return {
        "phone": phone[0] if phone else "Not found",
        "email": email[0] if email else "Not found"
    }

def calculate_similarity(model, job_text, resume_text):
    # Embedding and similarity calculation
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(job_embedding, resume_embedding).item()
    return {"similarity_score": round(similarity, 2)}

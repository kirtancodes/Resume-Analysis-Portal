import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# --- Step 1: Read PDF and Extract Text ---
def read_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    preprocess_resume(text)
    return text


# --- Step 2: Extract Fields from Text ---
def extract_resume_fields(text):
    """
    Extract specific fields from the resume text.
    """
    personal_details = extract_personal_details(text)
    qualifications = extract_education(text)
    skills = extract_skills(text)
    experience = extract_experience(text)
    projects = extract_projects(text)
    certifications = extract_certifications(text)

    return {
        'personal_details': personal_details,
        'qualifications': qualifications,
        'skills': skills,
        'experience': experience,
        'projects': projects,
        'certifications': certifications,
    }


# --- Helper Field Extraction Functions ---
def extract_personal_details(text):
    details = {'phone_numbers': [], 'emails': []}
    phone_numbers = re.findall(r'\+91-\d{10}', text)
    details['phone_numbers'] = phone_numbers
    emails = re.findall(r'\S+@\S+', text)
    details['emails'] = emails
    return details


def extract_education(text):
    education = []
    degree_keywords = ['B.Tech', 'M.Tech', 'B.Sc', 'M.Sc', 'Ph.D']
    for line in text.split("\n"):
        if any(degree in line for degree in degree_keywords):
            education.append(line.strip())
    return education


def extract_skills(text):
    skills = []
    skill_keywords = ["Python", "Java", "JavaScript", "ReactJS", "Machine Learning", "SQL", "HTML", "CSS","yurrffgfb7","pandas"]
    for keyword in skill_keywords:
        if keyword in text:
            skills.append(keyword)
    skills.sort()
    return skills


def extract_experience(text):
    experience = []
    for line in text.split("\n"):
        if 'intern' in line.lower() or 'worked as' in line.lower():
            experience.append(line.strip())
    return experience


def extract_projects(text):
    projects = []
    for line in text.split("\n"):
        if 'project' in line.lower():
            projects.append(line.strip())
    return projects


def extract_certifications(text):
    certifications = []
    for line in text.split("\n"):
        if 'certificate' in line.lower() or 'completed' in line.lower():
            certifications.append(line.strip())
    return certifications


# --- Step 3: Preprocess Resume Text ---
def preprocess_resume(raw_text):
    """
    Preprocess raw text to extract fields and combine them into a single string.
    """
    # Step 1: Extract structured fields from the text
    fields = extract_resume_fields(raw_text)

    # Step 2: Combine extracted fields into a unified string
    combined_text = []

    # Add personal details
    if fields['personal_details']['phone_numbers']:
        combined_text.append(f"Phone Numbers: {', '.join(fields['personal_details']['phone_numbers'])}")
    if fields['personal_details']['emails']:
        combined_text.append(f"Emails: {', '.join(fields['personal_details']['emails'])}")

    # Add qualifications
    if fields['qualifications']:
        combined_text.append(f"Qualifications: {', '.join(fields['qualifications'])}")

    # Add skills
    if fields['skills']:
        combined_text.append(f"Skills: {', '.join(fields['skills'])}")

    # Add experience
    if fields['experience']:
        combined_text.append(f"Experience: {', '.join(fields['experience'])}")

    # Add projects
    if fields['projects']:
        combined_text.append(f"Projects: {', '.join(fields['projects'])}")

    # Add certifications
    if fields['certifications']:
        combined_text.append(f"Certifications: {', '.join(fields['certifications'])}")

    # Combine all parts into a single string
    return "\n".join(combined_text)


# --- Example Usage ---
if __name__ == "__main__":
    # Path to the resume PDF
    resume_pdf_path = "C:\\Users\\Yash\\Desktop\\App_with_UI\\presume.pdf"  # Replace with your resume file path

    # Step 1: Read PDF and extract raw text
    raw_resume_text = read_pdf(resume_pdf_path)

    # Step 2: Preprocess the raw text to extract and combine fields
    processed_resume_text = preprocess_resume(raw_resume_text)

    # Print the processed resume text
    print("Processed Resume Text:")
    print(processed_resume_text)

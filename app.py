from flask import Flask, render_template, request, redirect, url_for
import os
from scripts.resume_processor import read_pdf, calculate_similarity_with_all_jds
from scripts.jd_processor import read_jd_file
from scripts.db_utils import get_mongo_connection, save_analysis_result, get_all_results
from scripts.video_processor import process_video_for_analysis

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

# Connect to MongoDB
db = get_mongo_connection()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recruiter", methods=["GET", "POST"])
def recruiter_upload():
    if request.method == "POST":
        # Job Description Upload
        if "jd" in request.files and request.files["jd"].filename != "":
            jd_file = request.files["jd"]
            jd_path = os.path.join(app.config["UPLOAD_FOLDER"], jd_file.filename)
            jd_file.save(jd_path)

            jd_text = read_jd_file(jd_path)  # Extract text from JD file
            db.job_descriptions.insert_one({
                "file_path": jd_path,
                "name": jd_file.filename,
                "text": jd_text
            })
            
        if "video" in request.files and request.files["video"].filename != "":
            video_file = request.files["video"]
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file.filename)
            video_file.save(video_path)  # Save the video
            result = process_video_for_analysis(video_path)
            save_analysis_result(db, "video", result)

        return redirect(url_for("index"))

    return render_template("recruiter_upload.html")

@app.route("/user", methods=["GET", "POST"])
def user_upload():
    if request.method == "POST":
        # Resume Upload
        resume_file = request.files["resume"]
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
        resume_file.save(resume_path)

        # Extract resume text
        resume_text = read_pdf(resume_path)

        # Compare resume with all JDs and calculate max similarity
        jd_list = list(db.job_descriptions.find())
        print()
        print("length",len(jd_list))
        best_match = calculate_similarity_with_all_jds(jd_list, resume_text)
        passing = [ {
            "resume_file": resume_file.filename,
            "best_jd_name": each["jd_name"],
            "similarity_score": each["similarity_score"],
            "similarity_details": each["details"],
        }  for each in best_match ]
        print()
        print("Passing",len(passing))
        # Save the best match result
        save_analysis_result(db, "resume", passing)

        return redirect(url_for("results"))

    return render_template("user_upload.html")

@app.route("/results")
def results():
    analysis_results = get_all_results(db)
    video_res=list(db.video.find())
    return render_template("results.html", results=analysis_results,video_res=video_res)

if __name__ == "__main__":
    # Ensure the uploads folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)

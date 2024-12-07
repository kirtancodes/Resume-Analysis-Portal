from flask_pymongo import PyMongo

def init_db(app):
    app.config["MONGO_URI"] = "mongodb://localhost:27017/recruitment_app"
    return PyMongo(app)

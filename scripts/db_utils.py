from pymongo import MongoClient

def get_mongo_connection():
    """
    Establish connection to MongoDB.
    """
    client = MongoClient('mongodb://localhost:27017/')
    db = client['job_analysis_db']
    return db

def save_analysis_result(db, collection_name, result):
    """
    Save analysis result to MongoDB.
    """
    db[collection_name].insert_many(result)

def get_all_results(db):
    """
    Retrieve all results from the analysis.
    """
    return list(db['resume'].find().sort({"similarity_score":-1}))

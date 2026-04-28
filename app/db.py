from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")


def get_db():
    password = os.getenv("MONGO_PASSWORD")
    uri = f"mongodb+srv://fa2927:{password}@cluster0.jphivpd.mongodb.net/"
    client = MongoClient(uri)
    db = client["big_data"]
    return db

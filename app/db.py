from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")


def get_db():
    user = os.getenv("MONGO_USER", "fa2927")
    password = os.getenv("MONGO_PASSWORD")
    uri = f"mongodb+srv://{user}:{password}@cluster0.jphivpd.mongodb.net/"
    client = MongoClient(uri)
    return client["big_data"]

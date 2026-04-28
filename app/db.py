from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path
import ssl
import certifi

load_dotenv(Path(__file__).parent / ".env")


def get_db():
    """
    Connect to MongoDB Atlas using credentials from .env file
    Each team member can use their own credentials
    """
    # Get credentials from environment variables
    mongo_username = os.getenv("MONGO_USERNAME")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_cluster = os.getenv("MONGO_CLUSTER", "cluster0.jphivpd.mongodb.net")
    mongo_database = os.getenv("MONGO_DATABASE", "big_data")
    
    # Validate required credentials
    if not mongo_username or not mongo_password:
        raise ValueError(
            "MongoDB credentials not found! "
            "Please set MONGO_USERNAME and MONGO_PASSWORD in app/.env file"
        )
    
    # Build connection URI with SSL parameters
    uri = f"mongodb+srv://{mongo_username}:{mongo_password}@{mongo_cluster}/"
    
    # Connect to MongoDB with SSL/TLS configuration
    # This fixes SSL handshake issues with Python 3.13+
    try:
        client = MongoClient(
            uri,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        # Test connection
        client.admin.command('ping')
        db = client[mongo_database]
        return db
    except Exception as e:
        # If certifi approach fails, try with default SSL context
        try:
            client = MongoClient(
                uri,
                tls=True,
                tlsAllowInvalidCertificates=False,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            client.admin.command('ping')
            db = client[mongo_database]
            return db
        except:
            # Re-raise original error with helpful message
            raise Exception(
                f"MongoDB connection failed: {str(e)}\n\n"
                "Possible solutions:\n"
                "1. Install certifi: pip install certifi\n"
                "2. Check your credentials in app/.env\n"
                "3. Verify IP is whitelisted in MongoDB Atlas\n"
                "4. Try: pip install --upgrade pymongo certifi"
            )

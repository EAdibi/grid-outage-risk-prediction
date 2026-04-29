from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path
import ssl
import certifi
import streamlit as st

load_dotenv(Path(__file__).parent / ".env")


def get_db(use_cache_fallback=True):
    """
    Connect to MongoDB Atlas using credentials from .env file
    Each team member can use their own credentials
    """
    # Get credentials from environment variables
    # Support both MONGO_USERNAME and MONGO_USER for compatibility
    mongo_username = os.getenv("MONGO_USERNAME") or os.getenv("MONGO_USER", "fa2927")
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
            # Try cache fallback if enabled
            if use_cache_fallback:
                st.warning("⚠️ **MongoDB Unreachable** - Using cached data instead")
                st.info(
                    "**Why?** Your IP address may not be whitelisted in MongoDB Atlas.\n\n"
                    "**To fix:**\n"
                    "1. Go to MongoDB Atlas → Network Access\n"
                    "2. Add your current IP address\n"
                    "3. Or add `0.0.0.0/0` to allow all IPs (less secure)\n\n"
                    "**For now:** Dashboard will use cached data (may be outdated)"
                )
                
                try:
                    from cache_manager import get_cached_db, is_cache_valid
                    
                    # Check if cache is available
                    if is_cache_valid('training_data'):
                        return get_cached_db()
                    else:
                        st.error(
                            "❌ **No cached data available**\n\n"
                            "Please connect to MongoDB at least once to cache data:\n"
                            "1. Whitelist your IP in MongoDB Atlas\n"
                            "2. Run: `python scripts/cache_data.py`"
                        )
                        raise Exception("No cached data available and MongoDB unreachable")
                except ImportError:
                    pass
            
            # Re-raise original error with helpful message
            raise Exception(
                f"MongoDB connection failed: {str(e)}\n\n"
                "Possible solutions:\n"
                "1. Whitelist your IP in MongoDB Atlas (Network Access)\n"
                "2. Check your credentials in app/.env\n"
                "3. Install certifi: pip install certifi\n"
                "4. Try: pip install --upgrade pymongo certifi"
            )

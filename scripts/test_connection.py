#!/usr/bin/env python3
"""
Test MongoDB connection with detailed debugging
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path("app/.env"))

print("=" * 70)
print("MongoDB Connection Test")
print("=" * 70)

# Show what we're trying to connect with (hide password)
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER", "cluster0.jphivpd.mongodb.net")
database = os.getenv("MONGO_DATABASE", "big_data")

print(f"\n📋 Configuration:")
print(f"   Username: {username}")
print(f"   Password: {'*' * len(password) if password else 'NOT SET'}")
print(f"   Cluster:  {cluster}")
print(f"   Database: {database}")

if not username or not password:
    print("\n❌ ERROR: Username or password not set in app/.env")
    print("\nYour app/.env should look like:")
    print("MONGO_USERNAME=jp8081")
    print("MONGO_PASSWORD=your_actual_password")
    exit(1)

print(f"\n🔗 Connection URI:")
uri = f"mongodb+srv://{username}:{'*' * len(password)}@{cluster}/"
print(f"   {uri}")

print(f"\n🔌 Attempting connection...")

try:
    from pymongo import MongoClient
    import certifi
    
    # Try connection
    client = MongoClient(
        f"mongodb+srv://{username}:{password}@{cluster}/",
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000
    )
    
    # Test with ping
    client.admin.command('ping')
    print("✅ Connection successful!")
    
    # List databases
    print(f"\n📊 Available databases:")
    for db_name in client.list_database_names():
        print(f"   - {db_name}")
    
    # Check target database
    db = client[database]
    collections = db.list_collection_names()
    
    print(f"\n📁 Collections in '{database}':")
    for coll in collections:
        count = db[coll].count_documents({})
        print(f"   - {coll}: {count:,} documents")
    
    print(f"\n🎉 Everything works! You're ready to go.")
    
except Exception as e:
    error_msg = str(e)
    print(f"❌ Connection failed!")
    print(f"\nError: {error_msg}")
    
    if "bad auth" in error_msg or "authentication failed" in error_msg:
        print(f"\n🔍 Authentication Error - Possible causes:")
        print(f"   1. Wrong username - verify it's exactly: {username}")
        print(f"   2. Wrong password - check for typos, spaces, special characters")
        print(f"   3. User doesn't exist in MongoDB Atlas")
        print(f"   4. User doesn't have access to database: {database}")
        
        print(f"\n💡 How to fix:")
        print(f"   1. Go to MongoDB Atlas → Database Access")
        print(f"   2. Find user: {username}")
        print(f"   3. If not found, ask teammate to create it")
        print(f"   4. If found, click 'Edit' → 'Edit Password' → Set new password")
        print(f"   5. Update app/.env with correct password")
        
    elif "IP" in error_msg or "whitelist" in error_msg:
        print(f"\n🔍 IP Whitelist Error:")
        print(f"   1. Go to MongoDB Atlas → Network Access")
        print(f"   2. Click 'Add IP Address'")
        print(f"   3. Either add your current IP or use 0.0.0.0/0 (allows all)")
        
    else:
        print(f"\n🔍 Unknown error. Check:")
        print(f"   1. MongoDB Atlas cluster is running")
        print(f"   2. Internet connection is working")
        print(f"   3. Credentials are correct")

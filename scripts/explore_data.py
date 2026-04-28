"""
Quick script to explore MongoDB data structure
Run this to understand the data before building visualizations and models
"""

from app.db import get_db
import json

def explore_collections():
    db = get_db()
    
    print("=" * 70)
    print("MONGODB DATA EXPLORATION")
    print("=" * 70)
    
    collections = db.list_collection_names()
    print(f"\n📊 Available Collections: {len(collections)}")
    print(f"{collections}\n")
    
    for collection_name in collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        
        print(f"\n{'='*70}")
        print(f"Collection: {collection_name}")
        print(f"{'='*70}")
        print(f"Total Documents: {count:,}")
        
        if count > 0:
            sample = collection.find_one()
            print(f"\nSample Document:")
            print(json.dumps(sample, indent=2, default=str))
            
            print(f"\nFields:")
            if sample:
                for key in sample.keys():
                    value = sample[key]
                    value_type = type(value).__name__
                    print(f"  - {key}: {value_type}")
        
        print()

if __name__ == "__main__":
    try:
        explore_collections()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. app/.env file exists with MONGO_PASSWORD")
        print("2. MongoDB Atlas is accessible")
        print("3. IP address is whitelisted")

"""
Cache MongoDB Data for Offline Dashboard Use
Run this when MongoDB is accessible to cache data locally
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from db import get_db
from cache_manager import cache_mongodb_data

print("=" * 70)
print("CACHING MONGODB DATA FOR OFFLINE USE")
print("=" * 70)
print()

try:
    # Connect to MongoDB (without cache fallback)
    print("📡 Connecting to MongoDB...")
    db = get_db(use_cache_fallback=False)
    print("✅ Connected to MongoDB\n")
    
    # Cache all collections
    cache_mongodb_data(db)
    
    print("=" * 70)
    print("✅ CACHING COMPLETE!")
    print("=" * 70)
    print()
    print("Your dashboard will now work offline for 24 hours!")
    print("Run this script again to refresh the cache.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Please fix MongoDB connection issues first:")
    print("1. Whitelist your IP in MongoDB Atlas")
    print("2. Check credentials in app/.env")
    print("3. Ensure internet connection is working")

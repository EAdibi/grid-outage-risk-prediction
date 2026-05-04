"""
Cache Manager for MongoDB Data
Allows dashboard to work offline by caching data locally
"""

import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours


def get_cache_path(cache_name):
    """Get path for cache file"""
    return CACHE_DIR / f"{cache_name}.pkl"


def is_cache_valid(cache_name):
    """Check if cache exists and is not expired"""
    cache_path = get_cache_path(cache_name)
    
    if not cache_path.exists():
        return False
    
    # Check if cache is expired
    modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
    if datetime.now() - modified_time > timedelta(hours=CACHE_EXPIRY_HOURS):
        return False
    
    return True


def save_to_cache(cache_name, data):
    """Save data to cache"""
    cache_path = get_cache_path(cache_name)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Cached {cache_name} to {cache_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to cache {cache_name}: {e}")
        return False


def load_from_cache(cache_name):
    """Load data from cache"""
    cache_path = get_cache_path(cache_name)
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded {cache_name} from cache")
        return data
    except Exception as e:
        print(f"❌ Failed to load {cache_name} from cache: {e}")
        return None


def cache_mongodb_data(db):
    """Cache all MongoDB collections for offline use"""
    print("📦 Caching MongoDB data for offline use...")
    
    collections_to_cache = {
        'training_data': 'training_data',
        'predictions': 'predictions',
        'outages_sample': 'outages',
        'storm_events_sample': 'storm_events',
        'county_population': 'county_population'
    }
    
    for cache_name, collection_name in collections_to_cache.items():
        try:
            # Limit samples for large collections
            if 'sample' in cache_name:
                data = list(db[collection_name].find().limit(10000))
            else:
                data = list(db[collection_name].find())
            
            save_to_cache(cache_name, data)
            print(f"   ✅ Cached {len(data):,} records from {collection_name}")
        except Exception as e:
            print(f"   ❌ Failed to cache {collection_name}: {e}")
    
    print("✅ Caching complete!\n")


def get_cached_db():
    """Get a mock database object that uses cached data"""
    class CachedDB:
        def __init__(self):
            self.training_data = CachedCollection('training_data')
            self.predictions = CachedCollection('predictions')
            self.outages = CachedCollection('outages_sample')
            self.storm_events = CachedCollection('storm_events_sample')
            self.county_population = CachedCollection('county_population')
    
    class CachedCollection:
        def __init__(self, cache_name):
            self.cache_name = cache_name
            self._data = load_from_cache(cache_name) or []
        
        def find(self, query=None, projection=None):
            """Simulate MongoDB find()"""
            return CachedCursor(self._data)
        
        def aggregate(self, pipeline):
            """Simulate MongoDB aggregate() - basic support"""
            # For now, just return the data
            # You can add pipeline processing if needed
            return self._data
        
        def count_documents(self, query=None):
            """Simulate MongoDB count_documents()"""
            return len(self._data)
        
        def distinct(self, field):
            """Simulate MongoDB distinct()"""
            values = set()
            for doc in self._data:
                if field in doc:
                    values.add(doc[field])
            return list(values)
    
    class CachedCursor:
        def __init__(self, data):
            self._data = data
            self._limit = None
        
        def limit(self, n):
            self._limit = n
            return self
        
        def __iter__(self):
            if self._limit:
                return iter(self._data[:self._limit])
            return iter(self._data)
        
        def __list__(self):
            if self._limit:
                return self._data[:self._limit]
            return self._data
    
    return CachedDB()

#!/usr/bin/env python3
"""
Quick setup verification script
Run this to ensure everything is configured correctly
"""

import sys
import os
from pathlib import Path

def check_env_file():
    """Check if .env file exists and is configured"""
    print("🔍 Checking environment configuration...")
    
    env_path = Path("app/.env")
    template_path = Path("app/.env.template")
    
    if not env_path.exists():
        print("❌ app/.env not found")
        print(f"   → Copy from template: cp app/.env.template app/.env")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        if 'your_username_here' in content or 'your_password_here' in content:
            print("⚠️  app/.env needs MongoDB credentials")
            print("   → Edit app/.env and set MONGO_USERNAME and MONGO_PASSWORD")
            return False
    
    # Check if required variables are set
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")
    
    if not username or not password:
        print("⚠️  MONGO_USERNAME or MONGO_PASSWORD not set in app/.env")
        return False
    
    print("✅ Environment file configured")
    print(f"   Username: {username}")
    return True

def check_dependencies():
    """Check if key packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required = ['streamlit', 'pymongo', 'pandas', 'plotly', 'scikit-learn']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages: pip install -r requirements.txt")
        return False
    
    return True

def check_mongodb_connection():
    """Test MongoDB connection"""
    print("\n🔍 Testing MongoDB connection...")
    
    try:
        from app.db import get_db
        
        db = get_db()
        collections = db.list_collection_names()
        
        print(f"✅ Connected to MongoDB")
        print(f"   Collections: {', '.join(collections[:5])}{'...' if len(collections) > 5 else ''}")
        
        # Check key collections
        key_collections = ['outages', 'storm_events', 'grid_demand']
        for coll in key_collections:
            if coll in collections:
                count = db[coll].count_documents({})
                print(f"   - {coll}: {count:,} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n   Troubleshooting:")
        print("   1. Check MONGO_PASSWORD in app/.env")
        print("   2. Verify MongoDB Atlas cluster is running")
        print("   3. Ensure IP address is whitelisted")
        return False

def check_project_structure():
    """Verify project structure"""
    print("\n🔍 Checking project structure...")
    
    required_files = [
        'app/main.py',
        'app/db.py',
        'app/sections/map.py',
        'app/sections/top_counties.py',
        'app/sections/texas_2021.py',
        'app/sections/feature_importance.py',
        'app/sections/comparative_analysis.py',
        'ml_pipeline/feature_engineering.py',
        'ml_pipeline/model_training.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 70)
    print("  Grid Outage Risk Platform - Setup Verification")
    print("=" * 70)
    print()
    
    checks = [
        ("Environment Configuration", check_env_file),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("MongoDB Connection", check_mongodb_connection)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error during {name}: {e}")
            results[name] = False
    
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all(results.values()):
        print("\n🎉 All checks passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Explore data: python explore_data.py")
        print("  2. Launch dashboard: cd app && streamlit run main.py")
        print("  3. Train models: python ml_pipeline/feature_engineering.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

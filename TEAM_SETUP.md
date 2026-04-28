# Team Setup Guide

## For Each Team Member

### 1. Clone Repository

```bash
git clone <repository-url>
cd grid-outage-risk-prediction
```

### 2. Create Your Branch

```bash
# Create a branch with your name
git checkout -b <your-name>-work

# Examples:
# git checkout -b jithendra-visualization
# git checkout -b elina-data-ingestion
# git checkout -b seungeun-ml
# git checkout -b will-spark
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure MongoDB Credentials

**Important**: Each team member uses their own MongoDB Atlas credentials.

```bash
# Copy the template
cp app/.env.template app/.env

# Edit app/.env with YOUR credentials
```

Your `app/.env` should look like:

```bash
# MongoDB Atlas Configuration
MONGO_USERNAME=jp8081              # ← YOUR username
MONGO_PASSWORD=your_password       # ← YOUR password
MONGO_CLUSTER=cluster0.jphivpd.mongodb.net
MONGO_DATABASE=big_data
```

### 5. Get Your MongoDB Credentials

#### Option A: If you already have MongoDB Atlas access

1. Go to [MongoDB Atlas](https://cloud.mongodb.com/)
2. Sign in with your account
3. Click "Database Access" in left sidebar
4. Find your username (e.g., `jp8081`)
5. If you don't remember password, click "Edit" → "Edit Password"

#### Option B: If you need to be added to the organization

Ask the team member who set up MongoDB Atlas (likely `fa2927`) to:
1. Go to MongoDB Atlas
2. Click "Database Access"
3. Click "Add New Database User"
4. Create user with your NetID (e.g., `jp8081`)
5. Set password and share it with you
6. Grant "Read and write to any database" permission

### 6. Verify Setup

```bash
# Test MongoDB connection
python explore_data.py

# If successful, you'll see:
# ======================================================================
# MONGODB DATA EXPLORATION
# ======================================================================
# 📊 Available Collections: 9
# ['county_population', 'disasters', 'generators', ...]
```

### 7. Launch Dashboard

```bash
cd app
streamlit run main.py

# Dashboard will open at http://localhost:8501
```

## Team Workflow

### Daily Work

```bash
# 1. Pull latest from main
git checkout main
git pull origin main

# 2. Switch to your branch and merge updates
git checkout <your-branch>
git merge main

# 3. Do your work

# 4. Commit and push
git add .
git commit -m "Add: description of your changes"
git push origin <your-branch>

# 5. Create Pull Request on GitHub when ready
```

### Resolving Conflicts

If you get merge conflicts:

```bash
# See which files have conflicts
git status

# Open conflicted files and fix them
# Look for <<<<<<< HEAD markers

# After fixing
git add .
git commit -m "Resolve merge conflicts"
```

## Team Member Responsibilities

### Jithendra (jp8081)
- **Dashboard Visualizations**: New pages in `app/sections/`
- **ML Model Training**: `ml_pipeline/`
- **Branch**: `visualization-ml-jithendra`

### Elina (fa2927)
- **Data Ingestion**: ETL scripts (if applicable)
- **MongoDB Setup**: Database administration

### Seungeun (sl12190)
- **TBD**: Based on team discussion

### Will (wdc9645)
- **TBD**: Based on team discussion

## Shared Resources

### MongoDB Atlas
- **Cluster**: `cluster0.jphivpd.mongodb.net`
- **Database**: `big_data`
- **Collections**: 9 collections (outages, storm_events, grid_demand, etc.)
- **Access**: Each member has their own username/password

### GitHub Repository
- **Main Branch**: `main` (protected, requires PR)
- **Feature Branches**: Each member works on their own branch
- **Pull Requests**: Required for merging to main

## Files You Should NOT Modify

To avoid conflicts, each team member should avoid modifying:

### Everyone Avoids
- `app/db.py` - MongoDB connection (unless coordinated)
- `requirements.txt` - Dependencies (unless coordinated)
- `.gitignore` - Git configuration (unless coordinated)

### Jithendra Avoids
- `app/sections/map.py` - Teammate's work
- `app/sections/top_counties.py` - Teammate's work
- `app/sections/texas_2021.py` - Teammate's work

### Coordination Required
- `app/main.py` - If adding new pages, coordinate navigation changes
- `README.md` - Update together or via PR

## Troubleshooting

### MongoDB Authentication Failed

```
Error: bad auth : authentication failed
```

**Solutions**:
1. Check `MONGO_USERNAME` in `app/.env` is correct
2. Check `MONGO_PASSWORD` in `app/.env` is correct
3. Verify your IP is whitelisted in MongoDB Atlas:
   - Go to MongoDB Atlas → Network Access
   - Add your IP or use `0.0.0.0/0` for development

### Module Not Found

```
ModuleNotFoundError: No module named 'xyz'
```

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Git Merge Conflicts

```
CONFLICT (content): Merge conflict in <file>
```

**Solution**:
1. Open the conflicted file
2. Look for conflict markers:
   ```
   <<<<<<< HEAD
   your changes
   =======
   their changes
   >>>>>>> main
   ```
3. Edit to keep what you want
4. Remove conflict markers
5. Save file
6. `git add <file>`
7. `git commit -m "Resolve conflicts"`

### Dashboard Won't Start

```
Error: Address already in use
```

**Solution**:
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run main.py --server.port 8502
```

## Quick Commands Reference

```bash
# Check current branch
git branch

# Switch branches
git checkout <branch-name>

# See your changes
git status
git diff

# Commit workflow
git add .
git commit -m "message"
git push origin <your-branch>

# Update from main
git checkout main
git pull
git checkout <your-branch>
git merge main

# Test MongoDB
python explore_data.py

# Launch dashboard
cd app && streamlit run main.py

# Train models (Jithendra's work)
python ml_pipeline/feature_engineering.py
python ml_pipeline/model_training.py
```

## Getting Help

1. **MongoDB Issues**: Ask Elina (fa2927) - database admin
2. **Git Issues**: Ask any team member or check GitHub docs
3. **Python Issues**: Check error message, Google, or ask team
4. **Dashboard Issues**: Ask Jithendra (jp8081) - visualization lead

## Communication

- **Slack/Discord**: Daily updates and quick questions
- **GitHub Issues**: Track bugs and features
- **Pull Requests**: Code review and discussion
- **Team Meetings**: Weekly sync on progress

---

**Remember**: Always work on your own branch and create PRs for review before merging to main!

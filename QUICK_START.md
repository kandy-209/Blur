# Quick Start: Deploy to Railway in 5 Steps

## Prerequisites Checklist
- [ ] Git installed (download from https://git-scm.com/download/win)
- [ ] GitHub account created (https://github.com)
- [ ] PowerShell restarted after Git installation

---

## Step-by-Step (Copy & Paste)

### 1. Open PowerShell in your project folder
```powershell
cd "C:\Users\Laxmo\Modal Test\modal_trading"
```

### 2. Configure Git (first time only)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Create GitHub Repository
- Go to: https://github.com/new
- Name: `futures-dashboard` (or any name)
- Click "Create repository"
- **Copy the repository URL** (looks like: `https://github.com/username/repo.git`)

### 4. Push Your Code
```powershell
# Initialize Git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Connect to GitHub (REPLACE with YOUR GitHub URL from step 3)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git branch -M main
git push -u origin main
```

**When asked for username/password:**
- Username: Your GitHub username
- Password: Use a Personal Access Token (see below)

### 5. Deploy to Railway
1. Go to: https://railway.app
2. Login with GitHub
3. New Project → Deploy from GitHub repo
4. Select your repository
5. Done! Get your URL

---

## Creating GitHub Personal Access Token

If Git asks for a password:

1. GitHub.com → Your Profile → Settings
2. Developer settings → Personal access tokens → Tokens (classic)
3. Generate new token (classic)
4. Check "repo" scope
5. Generate → **Copy the token**
6. Use this token as your password (not your GitHub password!)

---

## That's It!

Your dashboard will be live at: `https://your-app.railway.app`



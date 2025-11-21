# Complete Git & GitHub Setup Guide for Railway Deployment

## Step 1: Install Git

### Option A: Download Git for Windows (Recommended)
1. Go to: https://git-scm.com/download/win
2. Download the installer (it will auto-detect 64-bit)
3. Run the installer:
   - Click "Next" through all screens
   - **Important:** Choose "Git from the command line and also from 3rd-party software"
   - Click "Next" → "Install"
4. **Restart your terminal/PowerShell** after installation

### Option B: Install via Winget (if you have it)
```powershell
winget install --id Git.Git -e --source winget
```

### Verify Installation
Open a NEW PowerShell window and run:
```powershell
git --version
```
You should see something like: `git version 2.xx.x`

---

## Step 2: Configure Git (One-time setup)

Open PowerShell and run these commands (replace with YOUR info):

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Example:
```powershell
git config --global user.name "John Doe"
git config --global user.email "john@example.com"
```

---

## Step 3: Create GitHub Account (if you don't have one)

1. Go to: https://github.com
2. Click "Sign up"
3. Create your account (it's free)
4. Verify your email

---

## Step 4: Create a New Repository on GitHub

1. **Login to GitHub** (https://github.com)
2. Click the **"+" icon** in the top right → **"New repository"**
3. Fill in:
   - **Repository name:** `futures-trading-dashboard` (or any name you like)
   - **Description:** "Futures Trading Dashboard with Dark Theme"
   - **Visibility:** Choose "Public" (free) or "Private"
   - **DO NOT** check "Add a README file" (we'll add our own)
   - **DO NOT** add .gitignore or license (we have our own)
4. Click **"Create repository"**

5. **Copy the repository URL** - You'll see a page with instructions. Copy the HTTPS URL:
   - It looks like: `https://github.com/yourusername/futures-trading-dashboard.git`
   - **SAVE THIS URL** - you'll need it in the next step!

---

## Step 5: Initialize Git in Your Project

Open PowerShell in your project folder (`C:\Users\Laxmo\Modal Test\modal_trading`) and run:

```powershell
# Navigate to your project (if not already there)
cd "C:\Users\Laxmo\Modal Test\modal_trading"

# Initialize Git repository
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Futures Trading Dashboard"
```

---

## Step 6: Connect to GitHub and Push

**Replace `YOUR_GITHUB_URL` with the URL you copied in Step 4:**

```powershell
# Add GitHub as remote (REPLACE with your actual GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Rename default branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example** (if your username is "johndoe" and repo is "futures-dashboard"):
```powershell
git remote add origin https://github.com/johndoe/futures-dashboard.git
git branch -M main
git push -u origin main
```

### If GitHub asks for authentication:
- **Username:** Your GitHub username
- **Password:** You'll need a **Personal Access Token** (see Step 7)

---

## Step 7: Create GitHub Personal Access Token (if needed)

If Git asks for a password, you need a token:

1. Go to GitHub → Click your **profile picture** (top right)
2. Click **"Settings"**
3. Scroll down → Click **"Developer settings"** (left sidebar)
4. Click **"Personal access tokens"** → **"Tokens (classic)"**
5. Click **"Generate new token"** → **"Generate new token (classic)"**
6. Fill in:
   - **Note:** "Railway Deployment"
   - **Expiration:** Choose 90 days or "No expiration"
   - **Scopes:** Check **"repo"** (this gives full repository access)
7. Click **"Generate token"**
8. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
9. When Git asks for password, **paste this token** (not your GitHub password)

---

## Step 8: Deploy to Railway

Once your code is on GitHub:

1. Go to: https://railway.app
2. Click **"Login"** → Choose **"Login with GitHub"**
3. Authorize Railway to access your GitHub
4. Click **"New Project"**
5. Select **"Deploy from GitHub repo"**
6. Find and select your repository (`futures-trading-dashboard`)
7. Railway will automatically:
   - Detect it's a Python app
   - Install dependencies from `requirements.txt`
   - Run your app
   - Give you a public URL!

8. **That's it!** Your dashboard will be live at: `https://your-app.railway.app`

---

## Common Issues & Solutions

### "git is not recognized"
- **Solution:** Git isn't installed or terminal wasn't restarted
- Install Git (Step 1) and restart PowerShell

### "fatal: not a git repository"
- **Solution:** You're not in the project folder
- Run: `cd "C:\Users\Laxmo\Modal Test\modal_trading"`

### "remote origin already exists"
- **Solution:** Repository already connected
- Run: `git remote remove origin` then try again

### "Authentication failed"
- **Solution:** Use Personal Access Token (Step 7), not password

### "Permission denied"
- **Solution:** Make sure you're logged into GitHub
- Check: `git config --global user.name` and `git config --global user.email`

---

## Quick Reference Commands

```powershell
# Check Git status
git status

# See what files changed
git diff

# Add all files
git add .

# Commit changes
git commit -m "Your message here"

# Push to GitHub
git push

# See your commits
git log

# Check remote connection
git remote -v
```

---

## Need Help?

If you get stuck at any step, let me know which step and what error message you see!



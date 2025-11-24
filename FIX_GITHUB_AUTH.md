# Fix GitHub Authentication Error

You're getting "Permission denied (publickey)" because Git is trying to use SSH. Let's switch to HTTPS instead.

## Step 1: Check Your Remote URL

Run this command:
```bash
git remote -v
```

If you see `git@github.com:...` (SSH), we need to change it to HTTPS.

## Step 2: Change Remote to HTTPS

Replace the remote URL with HTTPS:

```bash
# Remove the current remote
git remote remove origin

# Add it back with HTTPS (REPLACE with YOUR GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Verify it's correct
git remote -v
```

You should now see `https://github.com/...` instead of `git@github.com:...`

## Step 3: Push Again

```bash
git push -u origin main
```

## Step 4: Authentication

When Git asks for credentials:
- **Username:** Your GitHub username
- **Password:** Use a Personal Access Token (see below)

## Create Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Fill in:
   - **Note:** "Railway Deployment"
   - **Expiration:** 90 days (or "No expiration")
   - **Scopes:** Check ✅ **"repo"** (this gives full repository access)
4. Click **"Generate token"**
5. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
6. When Git asks for password, **paste this token** (not your GitHub password)

---

## Quick Fix Commands

```bash
# Check current remote
git remote -v

# If it shows git@github.com, change to HTTPS:
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git push -u origin main
```



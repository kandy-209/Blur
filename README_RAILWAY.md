# Railway Deployment Guide

## Quick Deploy to Railway

### Option 1: Deploy via Railway CLI (Recommended)

1. **Install Railway CLI:**
   ```bash
   npm i -g @railway/cli
   ```

2. **Login to Railway:**
   ```bash
   railway login
   ```

3. **Initialize and Deploy:**
   ```bash
   railway init
   railway up
   ```

### Option 2: Deploy via GitHub (Easiest)

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to Railway.app:**
   - Visit https://railway.app
   - Sign up/login with GitHub
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect and deploy!

3. **That's it!** Railway will:
   - Install dependencies from `requirements.txt`
   - Run `python dashboard_modal.py`
   - Give you a public URL

### Environment Variables (Optional)

Railway will automatically:
- Set `PORT` environment variable
- Set `RAILWAY_ENVIRONMENT=production`

No additional configuration needed!

### Custom Domain (Optional)

1. In Railway dashboard, go to your project
2. Click on your service
3. Go to "Settings" → "Domains"
4. Add your custom domain

### Monitoring

- View logs: `railway logs` or in Railway dashboard
- View metrics: Railway dashboard shows CPU, memory, requests

## Cost

- **Free Trial:** $5 credit/month (enough for this app!)
- **After Trial:** ~$5-10/month for always-on service
- Much cheaper than Modal for always-on apps

## Troubleshooting

If the app doesn't start:
1. Check logs: `railway logs`
2. Ensure `requirements.txt` has all dependencies
3. Verify port is set correctly (Railway auto-sets PORT env var)

## Local Testing

Test locally with Railway's port:
```bash
PORT=8050 python dashboard_modal.py
```



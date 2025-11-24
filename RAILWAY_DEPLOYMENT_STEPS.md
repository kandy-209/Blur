# 🚂 Railway Deployment - Step by Step Guide

## Prerequisites
✅ Your code is already pushed to GitHub: `https://github.com/kandy-209/Blur.git`
✅ All deployment files are ready (`Procfile`, `railway.toml`, `requirements.txt`)

## Step 1: Create Railway Account

1. Go to **https://railway.app**
2. Click **"Start a New Project"** or **"Login"**
3. Sign in with your **GitHub account** (recommended)
   - This allows Railway to access your repositories automatically

## Step 2: Create New Project

1. After logging in, click **"New Project"** button (top right)
2. Select **"Deploy from GitHub repo"**
3. You'll see a list of your repositories
4. Find and click on **`Blur`** (or search for it)
5. Railway will ask for permission to access the repo - click **"Authorize"**

## Step 3: Configure Deployment

Railway will automatically:
- ✅ Detect it's a Python application
- ✅ Read `requirements.txt` for dependencies
- ✅ Use `Procfile` for the start command
- ✅ Set up the build process

**You don't need to change anything!** Just wait for the build to complete.

## Step 4: Wait for Build (2-5 minutes)

You'll see:
1. **Building** - Installing dependencies
2. **Deploying** - Starting your application
3. **Success** - Your app is live!

Watch the logs to see progress. You'll see messages like:
```
Installing dependencies...
Building application...
Starting web service...
```

## Step 5: Get Your Live URL

Once deployment succeeds:
1. Railway will show you a **public URL** like:
   ```
   https://your-app-name.up.railway.app
   ```
2. Click the **"Open"** button or copy the URL
3. Your dashboard is now live! 🎉

## Step 6: (Optional) Add Environment Variables

To enable enhanced features, add environment variables:

1. In Railway dashboard, click on your **project**
2. Go to **"Variables"** tab
3. Click **"New Variable"**
4. Add these one by one:

### Basic Auth (Recommended for Security)
```
DASH_ADMIN_USERNAME = admin
DASH_ADMIN_PASSWORD = your_secure_password_here
```

### Optional API Keys (for enhanced features)
```
NEWS_API_KEY = your_news_api_key
FRED_API_KEY = your_fred_api_key
OPENAI_API_KEY = your_openai_key
ANTHROPIC_API_KEY = your_anthropic_key
```

### Macro Research Settings (Optional)
```
MACRO_LOOKBACK = 90
MACRO_WEIGHTS = price_trend=0.3,volatility=0.25,macro_stress=0.25,growth_pulse=0.2
MACRO_SENTIMENT_WEIGHT = 0.05
```

**Note:** After adding variables, Railway will automatically redeploy your app.

## Step 7: Custom Domain (Optional)

1. In Railway dashboard → Your project → **Settings**
2. Scroll to **"Domains"** section
3. Click **"Generate Domain"** or add your own custom domain
4. Railway will provide SSL certificate automatically

## Troubleshooting

### ❌ Build Fails

**Check the logs:**
1. Go to your project in Railway
2. Click on the failed deployment
3. View the logs to see the error

**Common issues:**
- **Dependency errors**: Check if all packages in `requirements.txt` are valid
- **Python version**: Railway uses Python 3.11 by default (should work fine)
- **Memory issues**: Free tier has 512MB RAM limit

**Solution:** Check `requirements.txt` - all dependencies should be valid.

### ❌ App Won't Start

**Check:**
1. Port configuration - Your app uses `PORT` env var (Railway sets this automatically) ✅
2. Host binding - Your app binds to `0.0.0.0` ✅
3. Check logs for Python errors

### ❌ Charts Not Loading

**Possible causes:**
- yfinance rate limiting (temporary)
- Network issues
- Check browser console for errors

**Solution:** Wait a few minutes and refresh. yfinance may be rate-limited.

### ❌ 404 Errors

**Check:**
- Make sure you're visiting the root URL: `https://your-app.railway.app/`
- The app should serve at `/` (configured in `railway.toml`)

## Verification Checklist

After deployment, verify:

- [ ] Dashboard loads at the Railway URL
- [ ] Sidebar navigation works (Dashboard, Paper Trading, Analytics, Settings)
- [ ] Charts render correctly
- [ ] Data is updating (check the live indicator)
- [ ] Paper trading page loads
- [ ] No errors in browser console

## Monitoring Your App

Railway provides:
- **Logs**: Real-time application logs
- **Metrics**: CPU, Memory usage
- **Deployments**: History of all deployments
- **Settings**: Environment variables, domains, etc.

## Updating Your App

When you push new code to GitHub:
1. Railway automatically detects the push
2. Triggers a new deployment
3. Your app updates automatically!

**No manual steps needed!** 🎉

## Cost

Railway offers:
- **Free tier**: $5 credit/month (usually enough for small apps)
- **Hobby plan**: $5/month (if you need more resources)
- **Pro plan**: $20/month (for production apps)

Your dashboard should run fine on the free tier!

## Need More Help?

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check your deployment logs in Railway dashboard

---

**Your dashboard is ready to deploy! Follow steps 1-5 to go live in minutes! 🚀**


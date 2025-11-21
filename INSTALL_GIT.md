# How to Install Git on Windows

## Method 1: Download and Install (Recommended)

### Step 1: Download Git
1. Go to: **https://git-scm.com/download/win**
2. The download will start automatically (it detects 64-bit Windows)
3. Wait for the download to finish

### Step 2: Run the Installer
1. **Double-click** the downloaded file (usually in your Downloads folder)
   - File name: `Git-2.xx.x-64-bit.exe` (version number may vary)

2. **Installation Wizard:**
   - Click **"Next"** on the welcome screen
   - Click **"Next"** on the license screen
   - **Important:** On "Select Components" screen:
     - ✅ Check "Git from the command line and also from 3rd-party software"
     - ✅ Check "Associate .git* configuration files with the default text editor"
   - Click **"Next"**
   - Keep clicking **"Next"** through all remaining screens (defaults are fine)
   - Click **"Install"**
   - Wait for installation to complete
   - Click **"Finish"**

### Step 3: Restart PowerShell
1. **Close your current PowerShell window completely**
2. **Open a NEW PowerShell window**
3. Test Git installation:
   ```powershell
   git --version
   ```
   You should see: `git version 2.xx.x`

---

## Method 2: Install via Winget (If Available)

If you have Windows Package Manager (winget) installed:

```powershell
winget install --id Git.Git -e --source winget
```

Then restart PowerShell and test:
```powershell
git --version
```

---

## After Installation

Once Git is installed and you've restarted PowerShell, come back and we'll continue with:
1. Configuring Git
2. Creating GitHub repository
3. Pushing your code
4. Deploying to Railway

---

## Troubleshooting

**If `git --version` still doesn't work after restarting:**
1. Make sure you restarted PowerShell (not just refreshed)
2. Try restarting your computer
3. Check if Git was installed: Look for "Git Bash" in Start Menu
4. If Git Bash exists, Git is installed - the PATH might need a system restart



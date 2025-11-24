# Fix Your Remote URL

Your current remote URL is malformed. Here's how to fix it:

## Current (Wrong) URL:
```
https://github.com/kandy_209/git@github.com:kandy-209/Blur.git~
```

## Correct URL Should Be:
```
https://github.com/kandy-209/Blur.git
```

## Fix Commands:

```bash
# Remove the broken remote
git remote remove origin

# Add the correct remote URL
git remote add origin https://github.com/kandy-209/Blur.git

# Verify it's correct
git remote -v
```

You should now see:
```
origin  https://github.com/kandy-209/Blur.git (fetch)
origin  https://github.com/kandy-209/Blur.git (push)
```

Then push:
```bash
git push -u origin main
```



@echo off
echo Stage 1: Adding changes...
git add .
echo Stage 2: Committing changes...
git commit -m "Deployment Update: Added vercel.json, optimized .gitignore for Vercel, and finalized responsive UI fixes."
echo Stage 3: Pushing to GitHub...
git push
echo.
echo ==============================================
echo GITHUB PUSH COMPLETE
echo Next Step: Import this repo into Vercel.
echo ==============================================
echo.
pause

# Publishing this repo (5 minutes)

1. Create an empty public repo on GitHub named `content-intelligence-platform` (no README).
2. From this folder:
   ```bash
   git init && git add . && git commit -m "Content Intelligence Platform"
   git branch -M main
   git remote add origin https://github.com/iRoPad/content-intelligence-platform.git
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. Replace `iRoPad` in `README.md` (badge + live dashboard link) and push again.
5. Optional: **Settings → Secrets and variables → Actions → New secret** `ANTHROPIC_API_KEY`
   so CI runs the LLM tagger and the dashboard shows LLM accuracy instead of the rules baseline.
6. Pin the repo on your GitHub profile and add the live dashboard link to your resume and LinkedIn.

Delete this file once you're set up.

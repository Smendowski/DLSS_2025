Deep Learning Summer School

# 1. Environment setup
uv init --python=3.12.11
uv venv --python=3.12.11
source .venv/bin/activate
uv add -r requirements.txt
git remote add origin https://github.com/Smendowski/DLSS_2025.git
git branch -M main
git add .
git commit -m "Initial commit"
git push -u origin main

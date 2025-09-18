Deep Learning Summer School

# 1. Environment setup
uv init --python=3.12
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv add tensorflow-metal tensorflow-macos
git remote add origin https://github.com/Smendowski/DLSS_2025.git
git branch -M main
git add .
git commit -m "Initial commit"
git push -u origin main

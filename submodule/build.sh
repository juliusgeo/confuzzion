set -ex
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
python -m pip install --ignore-installed -e .
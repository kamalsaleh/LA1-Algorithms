#!/bin/bash

# This is a comment
echo "Running the application..."

# git pull
git pull origin master

# create vitual environment if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# activate virtual environment
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# navigate in your browser to http://localhost:5001
echo "Application is running on http://localhost:5001"

# run the application
python app.py
#!/bin/bash

if [ -d ".venv" ]
then
    source .venv/Scripts/activate
    pip install -r requirements.txt
    python wsgi.py
else
    python -m virtualenv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python wsgi.py
fi

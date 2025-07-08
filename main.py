"""
Root-level main.py for Render compatibility
Imports the actual FastAPI app from app/main.py
"""

from app.main import api as app

# This allows uvicorn main:app to work on Render
# while keeping our clean app/ structure

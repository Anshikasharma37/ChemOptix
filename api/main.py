
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes.predict import router as predict_router
from api.routes.optimize import router as optimize_router

load_dotenv()

# App 
app = FastAPI(
    title="ChemOptix API",
    description=(
        "AI-powered Gas Turbine yield and emission prediction with Gemini optimization. "
        "Frontend: Streamlit (streamlit_app.py)"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict_router)
app.include_router(optimize_router)


# Health check (used by Render)
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ChemOptix"}


# Root — redirect users to docs since UI is on Streamlit 
@app.get("/")
async def root():
    return {
        "message": "ChemOptix API is running.",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "single_predict": "POST /api/predict",
            "batch_predict":  "POST /api/predict/batch",
            "optimize":       "POST /api/optimize",
        }
    }

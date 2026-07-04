"""
ChemOptix — Gemini Optimization Route
Calls Gemini 1.5 Flash directly via REST API (no SDK version issues).
POST /api/optimize — returns structured suggestions for turbine parameters.
"""

import os
import json
import requests as http_requests
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from api.schemas import ProcessInput, PredictionResult, GeminiSuggestion

load_dotenv()

router = APIRouter(prefix="/api", tags=["Optimization"])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key={key}"
)


def _build_prompt(inputs: ProcessInput, predictions: PredictionResult) -> str:
    return f"""You are an expert industrial gas turbine optimization engineer.

A turbine operated under these conditions:
- Ambient Temperature (AT): {inputs.AT}°C
- Ambient Pressure (AP): {inputs.AP} mbar
- Ambient Humidity (AH): {inputs.AH}%
- Air Filter Differential Pressure (AFDP): {inputs.AFDP} mbar
- Gas Turbine Exhaust Pressure (GTEP): {inputs.GTEP} mbar
- Turbine Inlet Temperature (TIT): {inputs.TIT}°C
- Turbine After Temperature (TAT): {inputs.TAT}°C
- Compressor Discharge Pressure (CDP): {inputs.CDP} bar

The ML model predicted:
- Turbine Energy Yield (TEY): {predictions.TEY:.2f} MWH
- CO Emissions: {predictions.CO:.4f} mg/m³
- NOX Emissions: {predictions.NOX:.2f} mg/m³

Typical benchmarks for this turbine class:
- Good TEY: >140 MWH
- Acceptable CO: <2.0 mg/m³
- Acceptable NOX: <65 mg/m³

Analyze these results and provide optimization recommendations.
Respond ONLY with valid JSON in this exact format:
{{
  "summary": "One-sentence summary of the turbine current status",
  "suggestions": [
    "Actionable suggestion 1",
    "Actionable suggestion 2",
    "Actionable suggestion 3",
    "Actionable suggestion 4"
  ],
  "risk_level": "low"
}}

Risk level must be exactly one of: "low", "medium", or "high".
Base risk on: TEY below benchmark = medium/high risk; CO or NOX above threshold = high risk.
Keep each suggestion specific, technical, and actionable.
"""


def _rule_based_fallback(inputs: ProcessInput, predictions: PredictionResult) -> GeminiSuggestion:
    """Rule-based suggestions when Gemini is unavailable."""
    risk = "low"
    if predictions.TEY < 120 or predictions.CO > 5.0 or predictions.NOX > 90:
        risk = "high"
    elif predictions.TEY < 140 or predictions.CO > 2.0 or predictions.NOX > 65:
        risk = "medium"

    return GeminiSuggestion(
        summary=f"TEY at {predictions.TEY:.1f} MWH, CO at {predictions.CO:.3f} mg/m³, NOX at {predictions.NOX:.1f} mg/m³.",
        suggestions=[
            f"TIT is {inputs.TIT}°C — increasing toward 1095°C may improve energy yield.",
            f"CO at {predictions.CO:.3f} mg/m³ — monitor combustion mixture ratio and fuel quality.",
            f"NOX at {predictions.NOX:.1f} mg/m³ — consider water or steam injection to reduce emissions.",
            f"AFDP at {inputs.AFDP} mbar — schedule air filter inspection if above 4.5 mbar.",
        ],
        risk_level=risk,
    )


@router.post("/optimize", response_model=GeminiSuggestion)
async def get_optimization(inputs: ProcessInput, predictions: PredictionResult):
    """
    Call Gemini 1.5 Flash via direct REST API for turbine optimization suggestions.
    Falls back to rule-based suggestions if Gemini is unavailable.
    """
    if not GEMINI_API_KEY:
        return _rule_based_fallback(inputs, predictions)

    prompt = _build_prompt(inputs, predictions)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 512,
        },
    }

    try:
        response = http_requests.post(
            GEMINI_URL.format(key=GEMINI_API_KEY),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Strip markdown code blocks if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return GeminiSuggestion(
            summary=parsed.get("summary", "Optimization analysis complete."),
            suggestions=parsed.get("suggestions", []),
            risk_level=parsed.get("risk_level", "medium"),
        )

    except json.JSONDecodeError:
        return _rule_based_fallback(inputs, predictions)
    except http_requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        return GeminiSuggestion(
            summary=f"Gemini HTTP error {status} — check API key in Render environment variables.",
            suggestions=["Ensure GEMINI_API_KEY is a valid Google AI Studio key starting with AIzaSy..."],
            risk_level="medium",
        )
    except Exception as e:
        error_msg = str(e)
        # Never expose the API key in error messages
        if GEMINI_API_KEY and GEMINI_API_KEY in error_msg:
            error_msg = error_msg.replace(GEMINI_API_KEY, "***")
        return GeminiSuggestion(
            summary=f"Gemini error: {error_msg}",
            suggestions=["Check Render logs for details."],
            risk_level="medium",
        )


import os
import json
import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv

from api.schemas import ProcessInput, PredictionResult, GeminiSuggestion

load_dotenv()

router = APIRouter(prefix="/api", tags=["Optimization"])

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


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
  "summary": "One-sentence summary of the turbine's current status",
  "suggestions": [
    "Actionable suggestion 1",
    "Actionable suggestion 2",
    "Actionable suggestion 3",
    "Actionable suggestion 4"
  ],
  "risk_level": "low" | "medium" | "high"
}}

Base risk on: TEY below benchmark = medium/high risk; CO or NOX above threshold = high risk.
Keep each suggestion specific, technical, and actionable (mention exact parameter adjustments where possible).
"""


@router.post("/optimize", response_model=GeminiSuggestion)
async def get_optimization(inputs: ProcessInput, predictions: PredictionResult):
   
    if not GEMINI_API_KEY:
        # Return a fallback if no API key
        return GeminiSuggestion(
            summary="Gemini API key not configured. Using rule-based fallback.",
            suggestions=[
                f"TIT is {inputs.TIT}°C — increasing toward 1095°C may improve TEY.",
                f"CO at {predictions.CO:.3f} mg/m³ — monitor combustion mixture ratio.",
                f"NOX at {predictions.NOX:.2f} mg/m³ — consider water/steam injection.",
                f"Ensure air filter maintenance — AFDP of {inputs.AFDP} mbar indicates filter state.",
            ],
            risk_level="medium" if predictions.TEY < 140 or predictions.CO > 2.0 or predictions.NOX > 65 else "low",
        )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = _build_prompt(inputs, predictions)
        response = model.generate_content(prompt)

        # Parse JSON from Gemini response
        raw = response.text.strip()
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())

        return GeminiSuggestion(
            summary=data.get("summary", "Optimization analysis complete."),
            suggestions=data.get("suggestions", []),
            risk_level=data.get("risk_level", "medium"),
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Gemini returned malformed JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

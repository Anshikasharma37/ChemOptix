from pydantic import BaseModel, Field
from typing import Optional


class ProcessInput(BaseModel):
   

    AT: float = Field(..., description="Ambient Temperature (°C)", example=17.71, ge=-10, le=50)
    AP: float = Field(..., description="Ambient Pressure (mbar)", example=1013.07, ge=980, le=1050)
    AH: float = Field(..., description="Ambient Humidity (%)", example=77.87, ge=0, le=100)
    AFDP: float = Field(..., description="Air Filter Differential Pressure (mbar)", example=3.93, ge=2.0, le=8.0)
    GTEP: float = Field(..., description="Gas Turbine Exhaust Pressure (mbar)", example=25.56, ge=15, le=45)
    TIT: float = Field(..., description="Turbine Inlet Temperature (°C)", example=1081.43, ge=1000, le=1110)
    TAT: float = Field(..., description="Turbine After Temperature (°C)", example=546.16, ge=500, le=560)
    CDP: float = Field(..., description="Compressor Discharge Pressure (bar)", example=12.06, ge=9.0, le=16.0)
    year: Optional[int] = Field(2015, description="Measurement year (2011–2015)", ge=2011, le=2015)

    class Config:
        json_schema_extra = {
            "example": {
                "AT": 17.71,
                "AP": 1013.07,
                "AH": 77.87,
                "AFDP": 3.93,
                "GTEP": 25.56,
                "TIT": 1081.43,
                "TAT": 546.16,
                "CDP": 12.06,
                "year": 2015,
            }
        }


class PredictionResult(BaseModel):
   
    TEY: float = Field(..., description="Turbine Energy Yield (MWH)")
    CO: float = Field(..., description="CO Emission (mg/m³)")
    NOX: float = Field(..., description="NOX Emission (mg/m³)")


class GeminiSuggestion(BaseModel):
   
    summary: str = Field(..., description="Short optimization summary")
    suggestions: list[str] = Field(..., description="List of actionable suggestions")
    risk_level: str = Field(..., description="Overall risk assessment: low / medium / high")


class PredictionResponse(BaseModel):
    """Full API response: predictions + Gemini suggestions."""
    inputs: ProcessInput
    predictions: PredictionResult
    optimization: GeminiSuggestion
    status: str = "success"


class BatchPredictionResponse(BaseModel):
    """Single row result for batch/CSV prediction requests."""
    row_index: int = Field(..., description="Row number in uploaded CSV (1-based)")
    inputs: ProcessInput
    TEY: float = Field(..., description="Turbine Energy Yield (MWH)")
    CO: float = Field(..., description="CO Emission (mg/m³)")
    NOX: float = Field(..., description="NOX Emission (mg/m³)")
    status: str = Field("success", description="success or error message")

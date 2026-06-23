
import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List

from api.schemas import ProcessInput, PredictionResult, PredictionResponse, GeminiSuggestion, BatchPredictionResponse
from api.routes.optimize import get_optimization
from src.predict import predict

router = APIRouter(prefix="/api", tags=["Prediction"])

templates = Jinja2Templates(directory="templates")


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(inputs: ProcessInput):
   
    try:
        raw_preds = predict(inputs.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    predictions = PredictionResult(**raw_preds)

    try:
        optimization = await get_optimization(inputs, predictions)
    except HTTPException:
        # Fallback if Gemini fails
        optimization = GeminiSuggestion(
            summary="AI optimization temporarily unavailable.",
            suggestions=["Please retry for AI-powered suggestions."],
            risk_level="medium",
        )

    return PredictionResponse(
        inputs=inputs,
        predictions=predictions,
        optimization=optimization,
    )


@router.post("/predict/batch", response_model=List[BatchPredictionResponse])
async def predict_batch(inputs_list: List[ProcessInput]):
    
    if len(inputs_list) == 0:
        raise HTTPException(status_code=400, detail="No rows provided.")
    if len(inputs_list) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 rows per batch request.")

    results = []
    for i, row in enumerate(inputs_list):
        try:
            raw_preds = predict(row.model_dump())
            results.append(BatchPredictionResponse(
                row_index=i + 1,
                inputs=row,
                TEY=raw_preds["TEY"],
                CO=raw_preds["CO"],
                NOX=raw_preds["NOX"],
                status="success",
            ))
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            results.append(BatchPredictionResponse(
                row_index=i + 1,
                inputs=row,
                TEY=0.0, CO=0.0, NOX=0.0,
                status=f"error: {str(e)}",
            ))
    return results


@router.post("/predict/form", response_class=HTMLResponse)
async def predict_form(request: Request, inputs: ProcessInput):
    
    response = await predict_endpoint(inputs)
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "data": response.model_dump()},
    )

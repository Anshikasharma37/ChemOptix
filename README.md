# ChemOptix

**AI-Powered Gas Turbine Optimization Platform**

ChemOptix predicts gas turbine **energy yield (TEY)**, **CO emissions**, and **NOX emissions** from operational parameters, then delivers actionable optimization suggestions powered by **Google Gemini**.

Built with: FastAPI · XGBoost · Google Gemini · Deployed on Render

---

## Dataset

UCI Gas Turbine CO and NOx Emission Data Set (2011–2015)  
36,733 readings · 9 input features · 3 prediction targets

| Feature | Description |
|---------|-------------|
| AT | Ambient Temperature (°C) |
| AP | Ambient Pressure (mbar) |
| AH | Ambient Humidity (%) |
| AFDP | Air Filter Differential Pressure (mbar) |
| GTEP | Gas Turbine Exhaust Pressure (mbar) |
| TIT | Turbine Inlet Temperature (°C) |
| TAT | Turbine After Temperature (°C) |
| CDP | Compressor Discharge Pressure (bar) |

**Targets:** TEY (MWH) · CO (mg/m³) · NOX (mg/m³)

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" >> .env

# Place data files in data/raw/
# (gt_2011.csv .. gt_2015.csv)

# Train the model
python src/train.py

# Run the app
uvicorn api.main:app --reload
```

Open `http://localhost:8000`

---

## Project Structure

```
ChemOptix/
├── api/              # FastAPI app, routes, schemas
├── src/              # Preprocessing, training, inference
├── pipelines/        # ML pipeline definition
├── templates/        # Jinja2 HTML templates
├── static/           # CSS + JS
├── models/           # Trained model (git-ignored)
├── data/raw/         # Raw CSVs (git-ignored)
├── notebooks/        # EDA
├── render.yaml       # Render deployment config
└── Procfile
```

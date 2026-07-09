# ChemOptix — AI-Powered Gas Turbine Optimization Platform

[![Live Demo - Render UI](https://img.shields.io/badge/🚀_Live_Demo-Render_UI-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://chemoptix.onrender.com)
[![API Docs - FastAPI](https://img.shields.io/badge/⚡_API_Docs-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://chemoptix-api.onrender.com/docs)

---

## 🎯 Executive Summary & Purpose

### ⚠️ The Problem
In industrial power generation, operating gas turbines at maximum **Turbine Energy Yield (TEY)** while simultaneously keeping toxic environmental emissions—**Carbon Monoxide (CO)** and **Nitrogen Oxides (NOX)**—within strict regulatory limits is a complex, non-linear engineering challenge. 
Operators struggle to manually balance thermodynamic parameters (temperatures, pressures, airflow) in real-time. Modifying parameters to boost power output often spikes greenhouse gas emissions, risking severe regulatory penalties and equipment degradation.

### 💡 How ChemOptix Solves It
ChemOptix bridges **Machine Learning** and **Generative AI** into an automated industrial decision-support system. Instead of relying on manual trial-and-error, ChemOptix:
1. **Predicts in Real-Time:** Uses a high-precision **Multi-Output XGBoost Regression Model** (trained on 36,700+ operational records) to instantly predict energy yield (TEY), CO, and NOX emissions from 8 raw sensor measurements.
2. **Evaluates Risk:** Automatically benchmarks predictions against strict industrial safety thresholds (>140 MWH yield, <2.0 mg/m³ CO, <65 mg/m³ NOX) and assigns an operational risk status (**Low**, **Medium**, or **High**).
3. **Optimizes via Generative AI:** Feeds the thermodynamic state and ML predictions into **Google Gemini AI** using domain-specific prompt engineering. Gemini acts as an expert turbine engineer, generating actionable, step-by-step recommendations (e.g., adjusting Turbine Inlet Temperature or tuning air-fuel mixture ratios) to maximize efficiency while curbing emissions.

---

## 🚀 Core Features

* **Single Parameter Optimization:** Interactive Streamlit form with real-time AI-guided recommendations and risk classification.
* **Batch Inference Engine:** Upload CSV files of up to 500 turbine sensor readings at once for instantaneous bulk prediction and downloadable color-coded reporting.
* **Two-Service Cloud Architecture:** Decoupled REST API backend (**FastAPI**) and interactive UI frontend (**Streamlit**) deployed independently on **Render**.
* **Resilient Fallback System:** Built-in rule-based engineering recommendations ensure uninterrupted guidance even during network or LLM API outages.

---

## 📊 Dataset & Physics

Trained on the **UCI Gas Turbine CO and NOx Emission Data Set** (2011–2015), comprising **36,733 continuous readings** aggregated from real industrial power plants.

### Input Features (Sensor Telemetry)
| Symbol | Parameter Name | Unit | Engineering Relevance |
| :---: | :--- | :---: | :--- |
| **AT** | Ambient Temperature | °C | Modulates intake air density and compressor efficiency |
| **AP** | Ambient Pressure | mbar | Impacts total mass flow rate into the turbine |
| **AH** | Ambient Humidity | % | Affects heat capacity and combustion flame temperature |
| **AFDP** | Air Filter Differential Pressure | mbar | Indicates filter cleanliness and intake airflow resistance |
| **GTEP** | Gas Turbine Exhaust Pressure | mbar | Reflects back-pressure and turbine work extraction |
| **TIT** | Turbine Inlet Temperature | °C | **Primary driver of power output** and thermal efficiency |
| **TAT** | Turbine After Temperature | °C | Exhaust heat energy available for combined cycle recovery |
| **CDP** | Compressor Discharge Pressure | bar | Represents total compression ratio achieved before combustion |

### Prediction Targets & Benchmarks
* ⚡ **TEY (Turbine Energy Yield):** Target **> 140 MWH** *(Optimal power output)*
* ☁️ **CO (Carbon Monoxide):** Limit **< 2.0 mg/m³** *(Combustion efficiency indicator)*
* 🧪 **NOX (Nitrogen Oxides):** Limit **< 65 mg/m³** *(Environmental regulatory compliance)*

---

## 🛠️ Technology Stack

* **Machine Learning:** `XGBoost`, `Scikit-learn`, `Pandas`, `NumPy`, `Joblib`
* **Backend REST API:** `FastAPI`, `Pydantic`, `Uvicorn`, `Python-dotenv`
* **Frontend UI:** `Streamlit`
* **Generative AI:** `Google Gemini API` (REST v1beta with custom Prompt Engineering)
* **Cloud & DevOps:** `Render` (Two-Service Architecture), `Git`

---

## 💻 Local Setup & Running

```bash
# 1. Clone repository & install dependencies
git clone https://github.com/Anshikasharma37/ChemOptix.git
cd ChemOptix
pip install -r requirements.txt

# 2. Add your Google AI Studio API key
echo "GEMINI_API_KEY=AIzaSy_your_actual_key_here" >> .env

# 3. Start the FastAPI Backend Server (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 4. Start the Streamlit Frontend UI (Terminal 2)
streamlit run streamlit_app.py
```

* 🌐 **Frontend UI:** Open `http://localhost:8501`
* ⚡ **API Documentation:** Open `http://localhost:8000/docs`

---

## 📁 Project Architecture

```
ChemOptix/
├── api/
│   ├── routes/
│   │   ├── predict.py        # Single & batch ML inference endpoints
│   │   └── optimize.py       # Google Gemini AI optimization handler
│   ├── main.py               # FastAPI app initialization & CORS
│   └── schemas.py            # Pydantic data validation schemas
├── src/
│   ├── preprocessing.py      # Data cleaning, outlier filtering, scaling
│   ├── train.py              # XGBoost multi-output training script
│   └── predict.py            # Singleton ML model inference loader
├── pipelines/
│   └── ml_pipeline.py        # Scikit-learn feature/target pipeline definitions
├── models/
│   └── chemoptix_model.pkl   # Serialized production XGBoost model
├── notebooks/                # Exploratory Data Analysis (EDA)
├── streamlit_app.py          # Interactive Streamlit frontend UI
├── requirements.txt          # Pinned project dependencies
└── render.yaml               # Render cloud deployment blueprint
```

---

## 👩‍💻 Author & License

Developed by **Anshika Sharma** as an AI-powered industrial optimization project. Distributed under the MIT License.


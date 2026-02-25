from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import google.generativeai as genai
import pandas as pd
import io
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Cloudflare domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize external clients
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Use service role key to bypass RLS for server operations, or verify JWT directly
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    print("WARNING: Missing essential environment variables (Supabase or Gemini configs).")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
genai.configure(api_key=GEMINI_API_KEY)


def calculate_fingerprints(df: pd.DataFrame) -> dict:
    """Generate safe statistical representations of the data without exposing row data."""
    fingerprint = {
        "columns": [],
        "row_count": len(df)
    }
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        stats = {
            "name": col,
            "type": col_type,
            "null_count": int(df[col].isnull().sum())
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            stats.update({
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else 0
            })
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            top_values = df[col].value_counts().head(5)
            stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            stats["unique_count"] = int(df[col].nunique())
            
        fingerprint["columns"].append(stats)
        
    return fingerprint

@app.post("/api/analyze")
async def analyze_dataset(payload: dict):
    """
    Expects JSON mapping:
    {
        "file_path": "USER_ID/filename.csv",
        "access_token": "SUPABASE_JWT"
    }
    """
    file_path = payload.get("file_path")
    token = payload.get("access_token")
    
    if not file_path or not token:
        raise HTTPException(status_code=400, detail="Missing file_path or access_token")

    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
        
    # Optional: You can decode the JWT here to ensure the user actually matches the file_path
    # user = supabase.auth.get_user(token) ...
        
    try:
        # 1. Securely download the dataset from Supabase Storage
        res = supabase.storage.from_('app-files').download(file_path)
        
        # 2. Load into Pandas for purely safe, local statistical fingerprints
        if file_path.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(res))
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(res))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        fingerprint = calculate_fingerprints(df)

        # 3. Request insights from Gemini based ONLY on safe aggregated math. We demand strict JSON return.
        prompt = f"""
        You are NionData Server, an elite business operations analyst.
        Review the following statistical fingerprints derived from a private dataset.
        Generate rigorous, data-driven strategic insights based purely on these summary statistics.
        
        DATASET FINGERPRINT:
        {json.dumps(fingerprint, indent=2)}
        
        Respond ONLY with a raw JSON object matching this exact TypeScript interface:
        {{
            "smartAlerts": [{{ "id": string, "type": "warning" | "success" | "info" | "error", "title": string, "message": string, "metric": string }}],
            "rootCauses": [{{ "factor": string, "explanation": string, "impact": number (0-1) }}],
            "recommendations": [{{ "id": string, "title": string, "action": string, "impact": "high" | "medium" | "low", "priority": number, "estimatedROI": number, "savingsPotential": number }}],
            "productIntelligence": {{
              "topProducts": [{{ "name": string, "value": number }}] // Try to guess based on highest unique counts or max values
            }}
        }}
        
        Ensure your JSON uses double quotes and contains exactly this structure without markdown blocks (like ```json).
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        # Parse output safely
        output_text = response.text.replace('```json', '').replace('```', '').strip()
        analysis_result = json.loads(output_text)
        
        # 4. Return results safely to frontend React Application
        return {"status": "success", "analysis": analysis_result}
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

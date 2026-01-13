from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict
import io

app = FastAPI(title="Steel Loading Data Agent", description="Processes steel coil data and groups by destination.")

@app.post("/upload_excel/")
async def upload_excel(file: UploadFile = File(...)) -> Dict:
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        # Drop completely empty rows
        df.dropna(how="all", inplace=True)

        # Ensure required columns exist
        required_columns = ["到站", "毛重"]
        for col in required_columns:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"缺少必要字段 '{col}'"})

        # Drop rows where '到站' or '毛重' are null
        df = df[df["到站"].notnull() & df["毛重"].notnull()]

        # Remove header-duplicate rows (e.g. where values equal column names)
        expected_columns = ["材料号", "厚度", "宽度", "外径", "毛重", "净重"]
        df = df[~df.apply(lambda row: all(str(row[col]).strip() == col for col in expected_columns if col in row), axis=1)]

        # Group by '到站' and convert to structured JSON
        grouped = {}
        for station, group in df.groupby("到站"):
            products = group[expected_columns].dropna().to_dict(orient="records")
            if products:  # 只保留非空组
                grouped[station] = products

        return {"status": "success", "grouped_data": grouped}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Steel Loading Data Agent is running. Visit /docs to use the API."}

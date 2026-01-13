from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import uuid, os, json, io
from pathlib import Path
from Loading_agent import run_loading_process  # 确保存在且导入路径正确

app = FastAPI(title="Steel Loading Data Agent")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CURRENT_JSON_PATH = RESULTS_DIR / "current_grouped_data.json"


@app.post("/api/process_excel/")
async def process_excel(file: UploadFile = File(...), strategy: str = Form("vehicle_cost")):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        df.dropna(how="all", inplace=True)

        required_columns = ["到站", "毛重"]
        for col in required_columns:
            if col not in df.columns:
                return JSONResponse(status_code=400, content={
                    "status": "error",
                    "message": f"缺少字段 '{col}'",
                    "data": None
                })

        df = df[df["到站"].notnull() & df["毛重"].notnull()]
        expected_columns = ["材料号", "厚度", "宽度", "外径", "毛重", "净重"]
        df = df[~df.apply(lambda row: all(str(row[col]).strip() == col for col in expected_columns if col in row), axis=1)]

        grouped = {}
        for station, group in df.groupby("到站"):
            products = group[expected_columns].dropna().to_dict(orient="records")
            if products:
                grouped[station] = products

        file_id = uuid.uuid4().hex
        json_path = RESULTS_DIR / f"grouped_{file_id}.json"
        data_to_save = {"status": "success", "grouped_data": grouped}

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        with open(CURRENT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        pdf_path = RESULTS_DIR / f"report_{file_id}.pdf"
        result = run_loading_process(str(json_path), str(pdf_path), strategy=strategy)

        return {
            "status": "success",
            "message": f"文件处理成功，策略={strategy}",
            "data": {
                "pdf_url": f"/api/download_pdf/{pdf_path.name}",
                "json_url": f"/api/download_json/{json_path.name}",
                "summary": result["summary"]
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "data": None
        })


@app.post("/api/add_product/")
async def add_product(request: Request):
    try:
        data = await request.json()
        destination = data.get("destination")
        product_spec = data.get("product_spec")
        quantity_raw = data.get("quantity")

        if not destination or not product_spec or not quantity_raw:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "参数不完整",
                "data": None
            })

        try:
            quantity = int(quantity_raw)
            if quantity <= 0:
                raise ValueError
        except:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "数量必须为正整数",
                "data": None
            })

        parts = product_spec.strip().split('-')
        if len(parts) != 4:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "产品规格格式错误",
                "data": None
            })

        try:
            thickness, width, outer_diameter, gross_weight = map(float, parts)
        except:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "产品规格数值无效",
                "data": None
            })

        try:
            with open(CURRENT_JSON_PATH, encoding="utf-8") as f:
                current_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_data = {"status": "success", "grouped_data": {}}

        grouped_data = current_data.get("grouped_data", {})
        for _ in range(quantity):
            product = {
                "材料号": str(uuid.uuid4().hex[:12]),
                "厚度": thickness,
                "宽度": width,
                "外径": outer_diameter,
                "毛重": gross_weight,
                "净重": gross_weight
            }
            grouped_data.setdefault(destination, []).append(product)

        current_data["grouped_data"] = grouped_data
        with open(CURRENT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)

        return {
            "status": "success",
            "message": f"已添加 {quantity} 件产品至 {destination}",
            "data": {
                "grouped_data": grouped_data
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "data": None
        })


@app.post("/api/generate_from_current/")
async def generate_from_current(request: Request):
    try:
        body = await request.json()
        strategy = body.get("strategy", "vehicle_cost")

        if not CURRENT_JSON_PATH.exists():
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "当前没有数据，请先上传或添加产品",
                "data": None
            })

        file_id = uuid.uuid4().hex
        json_path = RESULTS_DIR / f"grouped_{file_id}.json"
        pdf_path = RESULTS_DIR / f"report_{file_id}.pdf"

        with open(CURRENT_JSON_PATH, encoding="utf-8") as f:
            current_data = json.load(f)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)

        result = run_loading_process(str(json_path), str(pdf_path), strategy=strategy)

        return {
            "status": "success",
            "message": f"基于当前数据生成成功，策略={strategy}",
            "data": {
                "pdf_url": f"/api/download_pdf/{pdf_path.name}",
                "json_url": f"/api/download_json/{json_path.name}",
                "summary": result["summary"]
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "data": None
        })


@app.get("/api/download_pdf/{filename}")
async def download_pdf(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={
            "status": "error",
            "message": "PDF 不存在",
            "data": None
        })
    return FileResponse(path, filename=filename, media_type="application/pdf")


@app.get("/api/download_json/{filename}")
async def download_json(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={
            "status": "error",
            "message": "JSON 不存在",
            "data": None
        })
    return FileResponse(path, filename=filename, media_type="application/json")


@app.get("/api")
async def root():
    return {"status": "success", "message": "Steel Loading API is running. Visit /docs to test.", "data": None}

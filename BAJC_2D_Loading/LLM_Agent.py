from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import openai
import os
import dotenv


# ✅ 可选：加载 .env 文件中的 OPENAI_API_KEY（你也可以手动设置下面那一行）
dotenv.load_dotenv()

# ✅ 推荐：使用环境变量方式设置 OpenAI 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 如果你只想测试，也可以取消注释下行直接写死密钥（不推荐用于生产）
# openai.api_key = "sk-你的密钥"

# 初始化 FastAPI 应用
app = FastAPI(title="LLM Agent", description="生成钢卷车皮分配方案")

# 定义钢卷产品的数据结构
class Product(BaseModel):
    材料号: str
    厚度: float
    宽度: float
    外径: float
    毛重: float
    净重: float

# 接收前端或 Data Agent 提交的数据结构
class AllocationRequest(BaseModel):
    destination: str
    products: List[Product]

# 接口：生成车皮分配方案
@app.post("/generate_plan/")
async def generate_plan(request: AllocationRequest):
    try:
        # 提取钢卷列表的简要信息（材料号与毛重）
        simplified_products = [{"材料号": p.材料号, "毛重": p.毛重} for p in request.products]

        # 构建中文 Prompt
        prompt = f"""你是一个专业的钢卷装载优化专家，请根据以下规则将钢卷分配到若干个车皮中：
1. 每个车皮最大载重为 65 吨；
2. 两个转向架重量差不超过 2 吨；
3. 同一排左右钢卷重量差不超过 1 吨；
4. 若中线不平衡，应满足对角线平衡（差值不超过 1 吨）。

到站：{request.destination}
钢卷清单如下：
{simplified_products}

请你生成一个清晰的车皮分配方案，指出每个车皮包含哪些钢卷，并指明左右/前后布局（可选），以最大限度满足上述条件。
"""

        # 调用 GPT 接口（建议 GPT-4）
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个擅长钢卷分配的工业智能专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        # 提取返回内容
        result = response['choices'][0]['message']['content']
        return {"status": "success", "plan": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

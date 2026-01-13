from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 DeepSeek 模型
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True).half().cuda()

def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试调用
if __name__ == "__main__":
    prompt = """
你是一位钢铁行业的装载专家，请根据以下约束条件为我分配钢卷到不同车皮中，并给出每个车皮内钢卷的具体位置安排：

【任务目标】
将一批钢卷分配到若干个车皮中，每个车皮的最大承载为 65 吨。要求在满足以下三类平衡约束条件的前提下，尽可能减少车皮数量，并优化装载平衡。

【车皮装载约束】
1. 载重限制：每个车皮总毛重不得超过 65 吨；
2. 轴重平衡：每个车皮有两个转向架，前后分段重量差不超过 2 吨；
3. 中线平衡：每一排左右两侧卷钢重量差不得超过 1 吨；
4. 对角线平衡：若中线平衡不满足，卷钢应沿对角线对称装载，对角线重量差 ≤ 1 吨。

【钢卷清单】
[
      {
        "材料号": "23957793801",
        "厚度": 0.8,
        "宽度": 1320,
        "外径": 1322,
        "毛重": 11.227,
        "净重": 11.13
      },
      {
        "材料号": "23957793802",
        "厚度": 0.8,
        "宽度": 1320,
        "外径": 1328,
        "毛重": 11.397,
        "净重": 11.3
      },
      {
        "材料号": "23350702802",
        "厚度": 0.71,
        "宽度": 1520,
        "外径": 1392,
        "毛重": 14.877,
        "净重": 14.78
      },
      {
        "材料号": "23350702801",
        "厚度": 0.71,
        "宽度": 1520,
        "外径": 1398,
        "毛重": 15.047,
        "净重": 14.95
      }
]

【输出格式】
请返回每个车皮的编号，以及对应分配的钢卷材料号、车皮内的位置（如前左、前右、后左、后右）和该车皮的总载重与是否满足所有约束。例如：

车皮1：
- 前左：222436401001（24.41 吨）
- 前右：222436401301（24.45 吨）
- 后左：222446900401（14.50 吨）
- 后右：222346202792（10.66 吨）
- 总毛重：64.02 吨，轴重差：1.01 吨，中线平衡：✓，对角线平衡：✓
"""

    result = generate_response(prompt)
    print("🧠 模型输出结果：\n", result)

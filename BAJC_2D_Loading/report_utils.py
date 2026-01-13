from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from collections import defaultdict
from typing import List, Dict
import os

def generate_pdf_and_summary(data: List[dict], output_pdf_path: str) -> Dict:
    """
    生成配载 PDF 报告，并返回结构化数据摘要
    :param data: 从 Excel 读取的产品列表，每项是一个 dict
    :param output_pdf_path: PDF 输出路径
    :return: 结构化摘要结果，用于前端展示
    """
    # 分组按目的地
    grouped_data = defaultdict(list)
    for item in data:
        dest = item.get("目的站") or item.get("destination_station") or "未知"
        grouped_data[dest].append(item)

    # 报告初始化
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 10)
    margin = 20 * mm
    y = height - margin

    summary = {
        "total_freight_cars": 0,
        "destinations": [],
    }

    # 每个目的地处理一次
    for i, (destination, items) in enumerate(grouped_data.items(), start=1):
        total_weight = 0.0
        car_count = 1
        car_weight = 0.0
        max_car_weight = 60.0  # 每车最大载重 60 吨
        car_summary = []

        c.drawString(margin, y, f"[{i}] Destination: {destination} | Product count: {len(items)}")
        y -= 12

        for idx, item in enumerate(items, start=1):
            try:
                weight = float(item.get("毛重", item.get("gross_weight", 0)) or 0)
            except ValueError:
                weight = 0

            total_weight += weight
            car_weight += weight

            if car_weight > max_car_weight:
                car_count += 1
                car_summary.append(car_weight - weight)
                car_weight = weight

            line = f"- #{idx} Material: {item.get('材料号', item.get('material_number', ''))} | Gross: {weight:.2f} t"
            c.drawString(margin + 10, y, line)
            y -= 12
            if y < 40 * mm:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - margin

        if car_weight > 0:
            car_summary.append(car_weight)

        avg_load_rate = sum(car_summary) / (60 * len(car_summary)) * 100 if car_summary else 0
        summary["total_freight_cars"] += len(car_summary)
        summary["destinations"].append({
            "destination": destination,
            "product_count": len(items),
            "freight_cars_used": len(car_summary),
            "average_loading_rate_percent": round(avg_load_rate, 2)
        })

        c.drawString(margin, y, f"↳ Freight Cars: {len(car_summary)} | Avg Loading Rate: {avg_load_rate:.2f}%")
        y -= 20
        if y < 40 * mm:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin

    c.save()
    return summary

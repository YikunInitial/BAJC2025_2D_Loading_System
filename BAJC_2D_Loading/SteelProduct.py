import json
from typing import List, Tuple
from collections import defaultdict
import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt

class Product:
    def __init__(self, material_number, storage_id, contract_number, product_name, destination_station,
                 receiving_company, thickness, width, outer_diameter, gross_weight, net_weight):
        # 材料号，库位号，合同号，货名，到站，收货单位，厚度，宽度，外径，毛重，净重
        self.material_number = material_number
        self.storage_id = storage_id
        self.contract_number = contract_number
        self.product_name = product_name
        self.destination_station = destination_station
        self.receiving_company = receiving_company
        self.thickness = thickness
        self.width = width
        self.outer_diameter = outer_diameter
        self.gross_weight = gross_weight
        self.net_weight = net_weight

    def __str__(self):
        return f"""
        Material Number: {self.material_number},
        Destination Station: {self.destination_station},
        Thickness: {self.thickness},
        Width: {self.width},
        Outer Diameter: {self.outer_diameter},
        Gross Weight: {self.gross_weight},
        Net Weight: {self.net_weight}.
        """

# Freight Car: length (长度)> the sum of (product.outer_diameter)
# width > the sum of (product.width)
# max_heavy > the sum of (product.gross_weight)

class FreightCar:
    # 车皮号，方案号，钢支架，钢材数量
    def __init__(self, freight_id, plan_id, steel_bracket, number, max_heavy=60.0):
        self.freight_id = freight_id
        self.plan_id = plan_id
        self.steel_bracket = steel_bracket
        self.number = number
        self.width = 3000.0 # 2800 ~ 3200 mm (2.8 - 3.2 m)
        self.length = 13000.0 # 13000 ~ 14000 mm (13 - 14 m)
        self.max_heavy = max_heavy # 60 ~ 70 t

    def __str__(self):
        return f"""
        Freight ID: {self.freight_id},
        Plan ID: {self.plan_id},
        Steel Bracket: {self.steel_bracket},
        Number of Product: {self.number}.
    """

# 用于生成带位置的产品信息
class PositionedGroup:
    def __init__(self, products: List[Product], position_x: float, position_y: float):
        self.products = products
        self.position_x = position_x
        self.position_y = position_y  # 0 = 左, 1 = 右, None = 单排

    def __str__(self):
        members = ', '.join(p.material_number for p in self.products)
        y_label = "Center" if self.position_y is None else ("Left" if self.position_y == 0 else "Right")
        return f"Group [{members}] at X={self.position_x:.1f}, Y={y_label}"


def load_products_from_json(filepath: str) -> List[Product]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    grouped_data = data.get("grouped_data", {})
    products = []

    for destination_station, items in grouped_data.items():
        for item in items:
            product = Product(
                material_number=item.get("材料号"),
                storage_id=None,  # Not available in JSON
                contract_number=None,  # Not available in JSON
                product_name=None,  # Not available in JSON
                destination_station=destination_station,
                receiving_company=None,  # Not available in JSON
                thickness=item.get("厚度"),
                width=item.get("宽度"),
                outer_diameter=item.get("外径"),
                gross_weight=item.get("毛重"),
                net_weight=item.get("净重")
            )
            products.append(product)

    return products

# 将产品按照重量降序排列
def sort_products_by_weight(products: List[Product]) -> List[Product]:
    """
    将产品按毛重（gross_weight）降序排序
    """
    return sorted(products, key=lambda p: p.gross_weight, reverse=True)

# 找到可行的可以并排的产品，满足外径相同，重量差低于1t，总宽度<车皮宽度
def find_parallel_pairs(products: List[Product], max_width=3000.0, max_weight_diff=1.0, force_same_outer_diameter=False):
    """
    根据宽度、重量差、外径相同优先生成并排组合组
    返回组结构：List[List[Product]]，每组是 [p1] 或 [p1, p2]
    """
    used = set()
    groups = []

    sorted_products = sort_products_by_weight(products)

    for i in range(len(sorted_products)):
        if i in used:
            continue
        p1 = sorted_products[i]
        candidate = None
        for j in range(i + 1, len(sorted_products)):
            if j in used:
                continue
            p2 = sorted_products[j]

            if abs(p1.gross_weight - p2.gross_weight) <= max_weight_diff and \
               (p1.width + p2.width <= max_width):

                if force_same_outer_diameter:
                    if p1.outer_diameter == p2.outer_diameter:
                        candidate = j
                        break
                else:
                    # 优先外径相同
                    if p1.outer_diameter == p2.outer_diameter:
                        candidate = j
                        break
                    elif candidate is None:
                        candidate = j  # 保留宽松组合做备选

        if candidate is not None:
            p2 = sorted_products[candidate]
            groups.append([p1, p2])
            used.add(i)
            used.add(candidate)
        else:
            groups.append([p1])
            used.add(i)

    return groups


# 预处理，将产品按照目的地分类

def group_products_by_destination(products: List[Product]) -> dict:
    grouped = defaultdict(list)
    for product in products:
        grouped[product.destination_station].append(product)
    return grouped

def compute_bogie_balance(groups: List[PositionedGroup], car_length: float) -> tuple:
    total_weight = sum(p.gross_weight for g in groups for p in g.products)
    moment = sum(p.gross_weight * g.position_x for g in groups for p in g.products)

    F2 = moment / car_length  # 后转向架（靠车尾）
    F1 = total_weight - F2    # 前转向架（靠车头）
    return round(F1, 2), round(F2, 2)

def is_within_bogie_balance(groups: List[PositionedGroup], car_length: float, tolerance=2.0) -> bool:
    F1, F2 = compute_bogie_balance(groups, car_length)
    return abs(F1 - F2) <= tolerance


def generate_initial_solution(products: List[Product], parallel_gap: float = 50.0) -> List[Tuple[FreightCar, List[PositionedGroup]]]:
    """
    - 保留并排 vs 单排组
    - 对称从中心放置
    - 载重 ≤ 60t；半车长（6500）内放置
    - 轴重平衡：前后转向架受力差 ≤ 2t，否则尝试贪心调整
    """
    groups = find_parallel_pairs(products)
    groups.sort(key=lambda g: sum(p.gross_weight for p in g), reverse=True)

    car_length = 13000.0
    half_len = car_length / 2
    max_weight = 65.0

    solution = []
    car_id = 1
    current_car = FreightCar(car_id, 1, "C1", 0)
    placed: List[PositionedGroup] = []
    used_weight = 0.0

    offset = 0.0
    direction = -1  # -1=left, +1=right

    for group in groups:
        # 组长度与组重量
        if len(group) == 2:
            g_len = group[0].outer_diameter + group[1].outer_diameter + parallel_gap
        else:
            g_len = group[0].outer_diameter
        g_wt = sum(p.gross_weight for p in group)

        # 计算下一组中心偏移
        next_offset = offset + g_len / 2

        # 如果超重或超半车长，则收车
        if used_weight + g_wt > max_weight or next_offset > half_len:
            # 先做轴重平衡检查
            if not is_within_bogie_balance(placed, car_length, tolerance=2.0):
                placed = smart_greedy_balance_positioning_nostack(placed, car_length)
            current_car.number = len(placed)
            solution.append((current_car, placed))
            # 重置新车
            car_id += 1
            current_car = FreightCar(car_id, 1, "C1", 0)
            placed = []
            used_weight = 0.0
            offset = 0.0
            direction = -1
            next_offset = g_len / 2

        # 计算本组中心 X
        center_x = half_len
        pos_x = center_x + direction * offset

        # 放置
        if len(group) == 2:
            p1, p2 = group
            placed.append(PositionedGroup([p1], pos_x, 0))
            placed.append(PositionedGroup([p2], pos_x, 1))
        else:
            placed.append(PositionedGroup(group, pos_x, None))

        # 更新累积
        used_weight += g_wt
        offset += g_len
        direction *= -1

    # 收尾最后一辆车
    if placed:
        if not is_within_bogie_balance(placed, car_length, tolerance=2.0):
            placed = smart_greedy_balance_positioning_nostack(placed, car_length)
        current_car.number = len(placed)
        solution.append((current_car, placed))

    return solution




# 奖励并排组，越对称代表越优化
def compute_symmetry_bonus(groups: List[PositionedGroup]) -> float:
    """
    奖励对称放置的并排组（Y=0 和 Y=1，且 X 相同，重量差小于1t）
    每个合格对称组奖励 0.5 分（从总得分中减去）
    """
    reward = 0.0
    used = set()
    for i, g1 in enumerate(groups):
        if i in used or g1.position_y not in (0, 1):
            continue
        for j, g2 in enumerate(groups):
            if j in used or i == j:
                continue
            if {g1.position_y, g2.position_y} == {0, 1} and g1.position_x == g2.position_x:
                w1 = sum(p.gross_weight for p in g1.products)
                w2 = sum(p.gross_weight for p in g2.products)
                if abs(w1 - w2) <= 1.0:
                    reward += 0.5
                    used.update({i, j})
                    break
    return reward


# 贪心启发式优化算法
def smart_greedy_balance_positioning_nostack(groups: List[PositionedGroup], car_length: float,
                                             tolerance: float = 2.0) -> List[PositionedGroup]:
    """
    调整轴重平衡：
    - 仅对所有 group 整体平移 X 保持相对位置不变
    - 保证并排对（Y=0/1）不拆分
    - 如前后受力差 <= tolerance(吨) 则无需调整
    - 如调整后任意 group 越界则保留原始位置
    """
    # 计算前/后转向架受力
    F1, F2 = compute_bogie_balance(groups, car_length)
    diff = abs(F1 - F2)
    if diff <= tolerance:
        return groups

    # 计算重心
    total_weight = sum(p.gross_weight for g in groups for p in g.products)
    com = sum(p.gross_weight * g.position_x for g in groups for p in g.products) / total_weight
    target_com = car_length / 2
    delta = target_com - com

    # 生成平移后的候选
    shifted = []
    for g in groups:
        new_x = g.position_x + delta
        # 检查越界
        d = g.products[0].outer_diameter
        if new_x - d/2 < 0 or new_x + d/2 > car_length:
            return groups  # 无法调整，返回原始
        shifted.append(PositionedGroup(g.products, new_x, g.position_y))

    return shifted


def perturb_solution(solution: List[Tuple[FreightCar, List[PositionedGroup]]],
                     split_prob: float = 0.3) -> List[Tuple[FreightCar, List[PositionedGroup]]]:
    """
    支持并排组拆分的扰动操作：
    - 有一定概率拆分并排组；
    - 随机交换两个单元（单排或并排）的位置；
    - 拓扑扰动后重新对每个车皮执行安全排布，避免重叠。
    """
    from copy import deepcopy

    def is_pair(g1: PositionedGroup, g2: PositionedGroup, tol=30.0):
        return g1.position_y in (0, 1) and g2.position_y in (0, 1) and \
               abs(g1.position_x - g2.position_x) < tol and \
               {g1.position_y, g2.position_y} == {0, 1}

    def reassign_positions(groups: List[PositionedGroup], car_length=13000, gap=50.0):
        center_x = car_length / 2
        left_cursor = center_x
        right_cursor = center_x
        center_cursor = center_x

        # 按重量排序放置更稳定
        sorted_groups = sorted(groups, key=lambda g: sum(p.gross_weight for p in g.products), reverse=True)

        for g in sorted_groups:
            d = g.products[0].outer_diameter
            if g.position_y == 0:
                g.position_x = left_cursor
                left_cursor -= (d + gap)
            elif g.position_y == 1:
                g.position_x = right_cursor
                right_cursor += (d + gap)
            else:
                g.position_x = center_cursor
                center_cursor += (d + gap)

        return sorted_groups

    new_solution = deepcopy(solution)

    candidate_indices = [i for i, (_, groups) in enumerate(new_solution) if len(groups) >= 2]
    if not candidate_indices:
        return new_solution

    car_idx = random.choice(candidate_indices)
    car, groups = new_solution[car_idx]

    # 分析组：找出原子组（单排或并排）
    used = set()
    group_blocks = []

    i = 0
    while i < len(groups):
        if i in used:
            i += 1
            continue

        g1 = groups[i]
        block = [g1]
        used.add(i)

        # 尝试组成并排
        for j in range(i + 1, len(groups)):
            if j in used:
                continue
            g2 = groups[j]
            if is_pair(g1, g2):
                block.append(g2)
                used.add(j)
                break

        group_blocks.append(block)
        i += 1

    # 如果有并排组，有概率拆开
    for block in group_blocks:
        if len(block) == 2 and random.random() < split_prob:
            # 拆开并设置为中间槽
            for g in block:
                g.position_y = None

    # 如果不足两个原子块，无法交换
    if len(group_blocks) < 2:
        return new_solution

    # 随机选两个 block 交换 x/y
    tries = 10
    for _ in range(tries):
        i, j = random.sample(range(len(group_blocks)), 2)
        for g1, g2 in zip(group_blocks[i], group_blocks[j]):
            g1.position_y, g2.position_y = g2.position_y, g1.position_y

    # 🔁 重排坐标
    new_groups = reassign_positions(groups)
    new_solution[car_idx] = (car, new_groups)
    return new_solution

def has_overlap(groups: List[PositionedGroup]) -> bool:
    seen = set()
    for g in groups:
        key = (round(g.position_x, 1), g.position_y)
        if key in seen:
            return True
        seen.add(key)
    return False


# 评估新的方案
def evaluate_solution(solution: List[Tuple[FreightCar, List[PositionedGroup]]]) -> float:
    """
    综合评价整个配载方案：
    - 车皮数量
    - 每车皮的轴重平衡
    - 装载利用率
    - 中线/对角线平衡（违反加罚）
    得分越低越好
    """
    total_score = 0.0
    for car, groups in solution:
        if has_overlap(groups):
            print(f"❌ 警告：车皮 {car.freight_id} 出现重叠组！强制惩罚")
            return float('inf')  # 重叠视为非法解

        # 载重
        total_weight = sum(p.gross_weight for g in groups for p in g.products)
        load_util = total_weight / car.max_heavy
        load_penalty = 1.0 - load_util  # 惩罚装载不足

        # 轴重差
        F1, F2 = compute_bogie_balance(groups, car.length)
        axle_diff = abs(F1 - F2)

        # 中线平衡
        _, mid_diff = check_midline_balance(groups)
        _, diag_diff = check_diagonal_balance(groups)

        mid_penalty = 2.0 if mid_diff > 1.0 else 0.0
        diag_penalty = 1.0 if diag_diff > 1.0 else 0.0
        # 对称奖励
        sym_bonus = compute_symmetry_bonus(groups)
        score = axle_diff + load_penalty + mid_penalty + diag_penalty - sym_bonus
        total_score += score

    # 加上车皮数量惩罚
    total_score += len(solution) * 5.0  # 可调整权重
    return total_score

def anneal(initial_solution: List[Tuple[FreightCar, List[PositionedGroup]]],
           initial_temp: float = 60.0,
           cooling_rate: float = 0.97,
           stopping_temp: float = 1e-5,
           max_iter: int = 1000) -> List[Tuple[FreightCar, List[PositionedGroup]]]:
    """
    模拟退火算法优化配载方案：
    - 以 initial_solution 为起点
    - 在邻域内随机扰动（perturb_solution）
    - 接受更优解或以概率接受较差解
    - 温度递减直至收敛
    """
    current = deepcopy(initial_solution)
    current_score = evaluate_solution(current)
    best = deepcopy(current)
    best_score = current_score
    temp = initial_temp
    iteration = 0

    print(f"开始模拟退火: 初始得分 {current_score:.2f}")

    while temp > stopping_temp and iteration < max_iter:
        neighbor = perturb_solution(current)
        neighbor_score = evaluate_solution(neighbor)

        # 是否接受新解？
        if neighbor_score < current_score:
            current = neighbor
            current_score = neighbor_score
        else:
            prob = math.exp((current_score - neighbor_score) / temp)
            if random.random() < prob:
                current = neighbor
                current_score = neighbor_score

        # 是否更新全局最优
        if current_score < best_score:
            best = deepcopy(current)
            best_score = current_score

        # 降温
        temp *= cooling_rate
        iteration += 1

        # 日志每 N 步打印一次
        if iteration % 20 == 0 or iteration == 1:
            print(f"Iter {iteration:3d} | Temp {temp:6.3f} | 当前得分 {current_score:.2f} | 最佳得分 {best_score:.2f}")

    print(f"✅ 模拟退火结束: 最佳得分 {best_score:.2f}，共迭代 {iteration} 轮")
    return best


# 计算中线平衡
def check_midline_balance(groups: List[PositionedGroup], tolerance=1.0) -> tuple:
    max_diff = 0.0
    for i in range(0, len(groups), 2):
        if i+1 < len(groups):
            g1 = groups[i]
            g2 = groups[i+1]
            if g1.position_x == g2.position_x and {g1.position_y, g2.position_y} == {0, 1}:
                w1 = sum(p.gross_weight for p in g1.products)
                w2 = sum(p.gross_weight for p in g2.products)
                diff = abs(w1 - w2)
                max_diff = max(max_diff, diff)
                if diff > tolerance:
                    return False, max_diff
    return True, max_diff


# 计算对角线平衡
def check_diagonal_balance(groups: List[PositionedGroup], tolerance=1.0) -> tuple:
    FL, FR, RL, RR = 0.0, 0.0, 0.0, 0.0
    mid = 13000 / 2

    for g in groups:
        weight = sum(p.gross_weight for p in g.products)
        front = g.position_x < mid
        left = g.position_y == 0
        right = g.position_y == 1

        if front and left:
            FL += weight
        elif front and right:
            FR += weight
        elif not front and left:
            RL += weight
        elif not front and right:
            RR += weight

    diff = abs((FL + RR) - (FR + RL))
    return diff <= tolerance, diff

# 输出配载结果： 总重量，总长度，转向架受力，中线平衡，对角平衡
def print_freight_summary(car: FreightCar, groups: List[PositionedGroup]):
    print(car)

    total_weight = sum(p.gross_weight for g in groups for p in g.products)
    total_length = sum(max(p.outer_diameter for p in g.products) for g in groups)
    F1, F2 = compute_bogie_balance(groups, car.length)
    midline_ok, mid_diff = check_midline_balance(groups)
    diagonal_ok, diag_diff = check_diagonal_balance(groups)

    print(f"  ➤ 总重量: {total_weight:.2f} t")
    print(f"  ➤ 总长度: {total_length:.0f} mm")
    print(f"  ➤ 转向架受力: F1 = {F1:.2f} t, F2 = {F2:.2f} t, 差值 = {abs(F1 - F2):.2f} t")
    print(f"  ➤ 中线平衡: {'✓' if midline_ok else '✗'} (最大差值: {mid_diff:.2f} t)")
    print(f"  ➤ 对角平衡: {'✓' if diagonal_ok else '✗'} (差值: {diag_diff:.2f} t)")

    for group in groups:
        print(f"    {group}")

    print("🔍 Debug - All group position info:")
    for g in groups:
        for p in g.products:
            print(f"Material: {p.material_number}, X: {g.position_x:.1f}, Y: {g.position_y}, D: {p.outer_diameter}")


# 绘制2D图
def plot_freight_car_2d_fixed(car, groups, save_path=None, gap=50.0):
    """
    精准二维配载图（并排产品上下对称，单件居中）：
    - X方向 position_x 为中心，宽度为 outer_diameter；
    - Y方向根据槽位调整，所有中心围绕1500展开；
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, car.length)
    ax.set_ylim(0, 3000)
    ax.set_title(f"Freight Car ID: {car.freight_id} - 2D Loading View")
    ax.set_xlabel("Position X (mm)")
    ax.set_ylabel("Position Y (mm)")
    ax.grid(True, linestyle='--', alpha=0.3)

    # 辅助线
    ax.axhline(1500, color='red', linestyle='--', alpha=0.5)
    ax.axvline(car.length / 2, color='gray', linestyle='--', alpha=0.3)

    for g in groups:
        p = g.products[0]
        width_x = p.outer_diameter
        height_y = p.width
        x_start = g.position_x - width_x / 2

        # ✅ 根据槽位决定 Y 起点
        if g.position_y == 0:        # Left 下侧
            y_center = 1500 - height_y / 2
        elif g.position_y == 1:      # Right 上侧
            y_center = 1500 + height_y / 2
        else:                        # Center 居中
            y_center = 1500

        y_start = y_center - height_y / 2  # 统一转换为左下角坐标

        label = f"{p.material_number}\n{p.gross_weight:.1f}t"

        rect = plt.Rectangle(
            (x_start, y_start),
            width_x,
            height_y,
            edgecolor='black',
            facecolor='skyblue',
            alpha=0.6
        )
        ax.add_patch(rect)

        ax.text(
            x_start + width_x / 2,
            y_center,
            label,
            ha='center',
            va='center',
            fontsize=7
        )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # 1. 加载数据
    product_list = load_products_from_json("data.json")
    grouped = group_products_by_destination(product_list)

    # 2. 对每个目的地分别处理
    for destination, products in grouped.items():
        print(f"\n📦 正在处理目的地：{destination} （产品数：{len(products)}）")

        # 2.1 并排分组信息
        print("🔍 find_parallel_pairs 分组结果:")
        groups = find_parallel_pairs(products)
        for idx, group in enumerate(groups, start=1):
            ids = ', '.join(p.material_number for p in group)
            tag = "并排" if len(group) == 2 else "单排"
            print(f"  Group {idx} ({tag}): {ids}")

        # 3. 初始方案生成
        initial_solution = generate_initial_solution(products, parallel_gap=50.0)
        print("\n🟦 初始方案:")
        for car, groups in initial_solution:
            print_freight_summary(car, groups)
            plot_freight_car_2d_fixed(car, groups, save_path=f"initial_{destination}_car_{car.freight_id}.png")

        # 4. 模拟退火优化
        optimized_solution = anneal(
            initial_solution,
            initial_temp=70.0,
            cooling_rate=0.97,
            stopping_temp=1e-5,
            max_iter=1000
        )

        print("\n🟩 优化后方案:")
        for car, groups in optimized_solution:
            print_freight_summary(car, groups)
            plot_freight_car_2d_fixed(car, groups, save_path=f"optimized_{destination}_car_{car.freight_id}.png")


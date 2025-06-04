from pymatgen.ext.matproj import MPRester

# 设置你的 Materials Project API 密钥
API_KEY = "Mq4eNAQa0GbXoPZf2GPJVKeX2GJnTKvL"

# 初始化 MPRester
with MPRester(API_KEY) as mpr:
    # 批量查询材料 ID 和相关属性
    material_ids = ["mp-1234", "mp-5678", "mp-91011"]  # 替换为实际的材料 ID 列表
    properties = ["material_id", "formula", "energy", "band_gap"]  # 指定需要的属性

    # 获取数据
    data = mpr.query({"material_id": {"$in": material_ids}}, properties)

# 输出结果
for entry in data:
    print(f"Material ID: {entry['material_id']}, Formula: {entry['formula']}, Energy: {entry['energy']}, Band Gap: {entry['band_gap']}")
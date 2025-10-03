import json
from tabulate import tabulate

def to_table(data):
    """ Convert invoice data to a formatted table string """
    # Hiển thị thông tin hóa đơn
    print(f"🛒 Hóa đơn từ: {data['SELLER']}")
    print(f"🕒 Thời gian: {data['TIMESTAMP']}")
    print("-" * 60)

    product_table = [
        [p["PRODUCT"], p["NUM"], f"{p['VALUE']:,} ₫"]
        for p in data["PRODUCTS"]
    ]

    print(tabulate(
        product_table,
        headers=["Sản phẩm", "Số lượng", "Thành tiền"],
        tablefmt="fancy_grid",
        stralign="left",
        numalign="right"
    ))

    print("-" * 60)
    print(f"💰 TỔNG CỘNG: {data['TOTAL_COST']:,} ₫")
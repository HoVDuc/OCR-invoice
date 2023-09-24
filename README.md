# OCR-invoice

## Introduction
OCR-invoice đây là đồ án tốt nghiệp của tôi, ứng dụng những gì đã học và tìm hiểu được để xây dựng một dự án về OCR cho việc nhận dạng và trích xuất thông tin hóa đơn bán hàng từ cửa hàng tiện lợi sau đó lưu lại dưới dạng Excel.

## Features
- Nhận dạng văn bản tiếng việt.
- Trích xuất thông tin từ hình ảnh hóa đơn.
- Lưu trữ thông tin dưới dạng Excel

## Installation:
### Requiremets:
- Python3.9
- Pytorch == 1.13.1
- GCC >= 4.9
- CUDA == 11.6

```
# Create conda env
conda create -n invoice_env python=3.9
conda activate invoice_env

# Clone this repo
git clone https://github.com/HoVDuc/OCR-invoice.git
cd OCR-invoice

# Install the necessary libraries
pip install -r requirements.txt

# Install pytorch with CUDA11.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu116
```

### Download weights:

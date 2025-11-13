# Label Region Extractor - Python Implementation

Chuyển đổi logic từ C# `LabelRegionExtractor.cs` sang Python để test/debug.

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Tạo virtual environment
py -m venv .venv

# Activate
.venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
```

### 2. Chuẩn bị ảnh test

Đặt ảnh test vào: `data/test_images/test.jpg`

### 3. Chạy detection

```powershell
python main.py
```

## 📁 Cấu trúc project

```
label-detector/
├── src/
│   ├── __init__.py
│   └── label_region_extractor.py    # Core logic (TẦNG 1 + TẦNG 2)
│
├── data/
│   ├── test_images/                 # Ảnh test (đặt ảnh vào đây)
│   └── results/                     # Output visualization
│
├── main.py                          # Script chạy test
├── requirements.txt                 # Dependencies
└── README.md                        # File này
```

## 🎯 Kiến trúc

### Hệ thống 3 tầng:

1. **TẦNG 1: Analysis**
   - `analyze_histogram()` - Tìm 2 peaks → separation
   - `analyze_edges()` - Canny edge detection → edge strength
   - `analyze_contrast()` - Standard deviation → contrast ratio
   - `analyze_frame()` - Tổng hợp → Final Score → phân loại Level

2. **TẦNG 2: Strategies**
   - **HIGH** (>0.45): Otsu adaptive threshold + Morphology
   - **MEDIUM** (0.25-0.45): Canny(30/100) + Loop candidates
   - **LOW** (<0.25): CLAHE + QR-First Geometry Inference

3. **Fallback Chain**
   - MEDIUM → HIGH (nếu fail)
   - LOW → MEDIUM → HIGH (2-level fallback)

## 🔧 Constants

```python
HIGH_THRESHOLD = 0.45       # Phân loại HIGH contrast
MEDIUM_THRESHOLD = 0.25     # Phân loại MEDIUM contrast
EDGE_MAX = 0.1              # Normalization cho edge strength

LABEL_WIDTH_RATIO = 4.0     # QR chiếm 1/4 chiều rộng label
LABEL_HEIGHT_RATIO = 3.0    # QR chiếm 1/3 chiều cao label
```

## 📊 Output

Script sẽ:
1. In ra console: Metrics, strategy selection, fallback chain
2. Hiển thị ảnh với visualization:
   - Label box (green)
   - QR points (red)
   - Strategy used
   - QR text
3. Lưu kết quả vào `data/results/output.jpg`

## 🧪 Testing với nhiều ảnh

Tạo script batch test:

```python
import cv2
from pathlib import Path
from src.label_region_extractor import detect_label_region

test_images = Path("data/test_images").glob("*.jpg")

for img_path in test_images:
    src = cv2.imread(str(img_path))
    result = detect_label_region(src)
    print(f"{img_path.name}: {result[5]}")  # Strategy used
```

## 📝 Notes

- QR points có thể là `None` nếu không detect được QR
- `qr_points_180` là tọa độ trong ROI cục bộ (Strategy HIGH/MEDIUM)
- `qr_points` là tọa độ trong ảnh gốc (toàn bộ strategies)
- Strategy LOW không có `qr_points_180` vì detect trên toàn ảnh

## 🔍 Troubleshooting

### Không detect được label

1. Kiểm tra metrics trong console → xem Level nào được chọn
2. Xem log chi tiết của từng strategy
3. Thử điều chỉnh constants (HIGH_THRESHOLD, MEDIUM_THRESHOLD)
4. Kiểm tra ảnh có QR code không (Strategy LOW cần QR)

### QR detection fail

- Thử tăng độ phân giải ảnh
- Kiểm tra QR có rõ ràng không (bị mờ, nghiêng quá nhiều)
- Strategy LOW apply CLAHE + histogram equalization để enhance

## 🚀 Performance

- **HIGH**: ~5-10ms
- **MEDIUM**: ~15-25ms  
- **LOW**: ~5-10ms

→ Trung bình: ~10ms (~100 FPS)

## 📚 Tham khảo

- Source code C#: `LabelRegionExtractor.cs`
- Kế hoạch triển khai: `PYTHON_IMPLEMENTATION_PLAN.md`

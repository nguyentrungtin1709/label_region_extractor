# Kế Hoạch Cập Nhật Logic Label Region Extractor

## Tổng Quan
Cập nhật lớn logic phân tích và chiến lược detection:
- **Loại bỏ**: Strategy MEDIUM và metric `analyze_edges`
- **Giữ lại**: Chỉ 2 strategies (HIGH và LOW)
- **Thay đổi**: Logic scoring, threshold method, và fallback chain

---

## 📋 Danh Sách Thay Đổi Chi Tiết

### 1. TẦNG 1: Analysis Methods

#### 1.1. `analyze_histogram()` - CẬP NHẬT
**Thay đổi:**
- Trả về **4 giá trị** thay vì 3: `(peak1, peak2, trough_pos, separation)`
- Thêm logic tính **trough position** (điểm có tần suất thấp nhất giữa 2 đỉnh)
- Trough = giá trị ngưỡng tối ưu cho Simple Thresholding

**Logic mới:**
```python
if peak1 < peak2:
    trough_pos = np.argmin(smoothed[peak1:peak2+1]) + peak1
else:
    trough_pos = 127  # Fallback
```

**Debug visualization:**
- Thêm đường thẳng đứng màu tím (`purple`) cho trough position
- Label: `Trough={trough_pos} (Threshold)`

---

#### 1.2. `analyze_edges()` - XÓA BỎ
**Hành động:**
- Xóa toàn bộ hàm `analyze_edges()`
- Xóa file debug: `debug_02_edges.png`

**Lý do:**
- Metric edge không còn được sử dụng trong công thức final_score mới

---

#### 1.3. `analyze_contrast()` - GIỮ NGUYÊN
**Không thay đổi:**
- Logic giữ nguyên
- Trả về: `(mean, stddev, contrast_ratio)`

---

#### 1.4. `analyze_frame()` - CẬP NHẬT LỚN
**Thay đổi:**

1. **Loại bỏ metric edge:**
   ```python
   # XÓA: edge_pixels, edge_strength = analyze_edges(gray)
   # XÓA: edge_strength_norm = min(edge_strength / EDGE_MAX, 1.0)
   ```

2. **Công thức final_score MỚI:**
   ```python
   # CŨ: separation×0.4 + edge_strength×0.3 + contrast_ratio×0.3
   # MỚI:
   final_score = (separation * 0.6) + (contrast_ratio * 0.4)
   ```

3. **Logic phân cấp MỚI:**
   ```python
   # CŨ: >0.45=High, 0.25-0.45=Medium, <0.25=Low
   # MỚI:
   if separation > 0 and final_score > 0.3:
       level = 'High'
   else:
       level = 'Low'
   ```

4. **Điều kiện quan trọng:**
   - `separation > 0`: Đảm bảo có 2 đỉnh rõ ràng (bimodal)
   - `final_score > 0.3`: Đảm bảo độ tách đủ mạnh

---

#### 1.5. `ContrastAnalysisResult` - CẬP NHẬT
**Thay đổi:**

```python
@dataclass
class ContrastAnalysisResult:
    level: str  # 'High' hoặc 'Low' (xóa 'Medium')
    final_score: float
    
    # XÓA: edge_strength, edge_pixel_count
    # THÊM: trough_position
    
    separation: float
    contrast_ratio: float
    
    peak1_position: int
    peak2_position: int
    trough_position: int  # ← MỚI: Ngưỡng cho HIGH strategy
    mean_intensity: float
    stddev_intensity: float
```

---

### 2. TẦNG 2: Strategy Methods

#### 2.1. `detect_with_high_contrast()` - CẬP NHẬT LỚN
**Thay đổi:**

1. **Thay Otsu bằng Simple Threshold:**
   ```python
   # CŨ:
   otsu_threshold, binary = cv2.threshold(
       gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
   )
   
   # MỚI:
   _, binary = cv2.threshold(
       gray, threshold_value, 255, cv2.THRESH_BINARY
   )
   ```

2. **Tham số mới:**
   - `threshold_value: int` - Nhận giá trị từ `analysis.trough_position`
   - Đây là điểm có tần suất thấp nhất giữa 2 đỉnh histogram

3. **Cập nhật print statement:**
   ```python
   print(f"  → Applied Simple Threshold: {threshold_value}")
   ```

4. **Giữ nguyên:**
   - Morphology: Open(3×3, 1 iter) → Close(3×3, 2 iters)
   - Find largest contour
   - QR verification

---

#### 2.2. `detect_with_medium_contrast()` - XÓA BỎ
**Hành động:**
- Xóa toàn bộ hàm `detect_with_medium_contrast()`
- Xóa tất cả file debug có prefix `medium_*`

**Lý do:**
- Qua thử nghiệm thực tế, strategy này không hiệu quả

---

#### 2.3. `detect_with_low_contrast()` - GIỮ NGUYÊN
**Không thay đổi:**
- Logic QR-First + Geometry Inference giữ nguyên
- Multi-method preprocessing (CLAHE → Histogram Equalization → Original BGR)

---

### 3. HÀM CHÍNH: `detect_label_region()`

#### 3.1. Routing Logic - CẬP NHẬT LỚN
**Chỉ còn 2 nhánh:**

##### **Nhánh 1: High Strategy**
```python
if analysis.level == 'High':
    print("🟢 Executing Strategy: HIGH CONTRAST (Primary)")
    
    # Thử HIGH với threshold = trough_position
    result = detect_with_high_contrast(
        src, gray_blurred, analysis.trough_position
    )
    
    if result[0] is not None:
        strategy_used = "HIGH"
    else:
        # FALLBACK: HIGH → LOW
        print("⚠️  HIGH failed, falling back to LOW strategy...")
        
        # Apply CLAHE cho LOW
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray_blurred)
        
        result = detect_with_low_contrast(src, gray_clahe)
        
        if result[0] is not None:
            strategy_used = "HIGH→LOW"
```

##### **Nhánh 2: Low Strategy**
```python
else:  # analysis.level == 'Low'
    print("🔴 Executing Strategy: LOW CONTRAST (Primary)")
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray_blurred)
    
    # Thử LOW
    result = detect_with_low_contrast(src, gray_clahe)
    
    if result[0] is not None:
        strategy_used = "LOW"
    else:
        # KHÔNG FALLBACK - Dừng lại
        print("⚠️  LOW failed. No fallback (Histogram not separable).")
        strategy_used = "FAILED"
```

---

#### 3.2. Fallback Chain - LOGIC MỚI
**Quy tắc:**

1. **High → Low**: ✅ Cho phép
   - **Lý do**: Nếu histogram có 2 đỉnh nhưng threshold không hiệu quả, có thể QR-first approach vẫn hoạt động

2. **Low → High**: ❌ KHÔNG cho phép
   - **Lý do**: Nếu histogram KHÔNG phân tách được (Low), thì không thể dùng threshold-based approach (High)
   - **Logic**: `separation == 0` hoặc `final_score <= 0.3` → không có trough hợp lệ

---

#### 3.3. Cập nhật Log Output
**Thay đổi:**

1. **Header log:**
   ```python
   print("║  🎯 Strategy Level:  {analysis.level:<10} (Primary)                 ║")
   print("║    • HIGH Threshold: {analysis.trough_position:<6} (Trough)         ║")
   ```

2. **Loại bỏ:**
   ```python
   # XÓA: Edge Strength log
   # XÓA: Edge Pixel Count log
   ```

---

### 4. Constants - CẬP NHẬT

#### 4.1. Xóa constants không dùng:
```python
# XÓA:
# HIGH_THRESHOLD = 0.45
# MEDIUM_THRESHOLD = 0.25
# EDGE_MAX = 0.1
```

#### 4.2. Giữ lại:
```python
# Expansion ratios cho LOW strategy
QR_VERTICAL_CENTER_UP = 1.0
QR_VERTICAL_CENTER_DOWN = 1.2
QR_HORIZONTAL_RIGHT = 0.2
QR_LEFT_EXPANSION = 3.5
PADDING_RATIO = 0.1

DEBUG_OUTPUT_DIR = "data/debug"
```

---

## 🎯 Lý Do Chọn Trough Position Làm Ngưỡng

### Phân Tích 3 Tùy Chọn:

#### ✅ Tùy chọn 1: Trough (Điểm thấp nhất giữa 2 đỉnh) - **ĐƯỢC CHỌN**
**Ưu điểm:**
- **Lý thuyết vững chắc**: Trong histogram bimodal, trough chính là ranh giới tự nhiên giữa 2 nhóm pixel
- **Tương thích với ảnh thực tế**: Nhìn `debug_01_histogram.png`, trough nằm chính xác giữa đỉnh nền (176) và đỉnh nhãn (213)
- **Optimal separation**: Tách nhãn ra khỏi nền với sai số tối thiểu

**Ví dụ từ ảnh:**
- Peak1=176 (nền)
- Peak2=213 (nhãn)
- Trough≈194 (điểm thấp nhất giữa 2 đỉnh)
- Nếu dùng 194 làm threshold → pixel < 194 là nền, pixel ≥ 194 là nhãn

---

#### ❌ Tùy chọn 2: Mean + StdDev
**Nhược điểm:**
- **Không phản ánh ranh giới**: `Mean+StdDev=199.2` nằm sâu trong đỉnh nhãn (213)
- **Cắt xén nhãn**: Pixel từ 176→199 của nhãn sẽ bị phân loại sai thành nền
- **Không tối ưu cho bimodal**: Metric này đo độ phân tán chung, không đo sự tách biệt

**Ví dụ thất bại:**
```
Histogram:      Nền (176)     Nhãn (213)
                   ▲              ▲
                  / \            / \
                 /   \          /   \
                /     \        /     \
               /       \______/       \
              0     176  194 199.2  213  255
                           ↑    ↑
                       Trough  Mean+σ (SAI)
```

---

#### ❌ Tùy chọn 3: Trung bình của Trough và Mean+StdDev
**Nhược điểm:**
- **Làm "ô nhiễm" giá trị tối ưu**: Kéo trough ra xa vị trí lý tưởng
- **Không có lợi ích**: Chỉ là타협 không cần thiết

---

## 📝 Checklist Thực Hiện

### Phase 1: Cập nhật Analysis Methods
- [ ] Cập nhật `analyze_histogram()`: Thêm trough calculation
- [ ] Xóa `analyze_edges()` và debug files
- [ ] Cập nhật `analyze_contrast()` (giữ nguyên nhưng check lại)
- [ ] Cập nhật `analyze_frame()`: Logic scoring mới
- [ ] Cập nhật `ContrastAnalysisResult` dataclass

### Phase 2: Cập nhật Strategy Methods
- [ ] Cập nhật `detect_with_high_contrast()`: Otsu → Simple Threshold
- [ ] Xóa `detect_with_medium_contrast()` và debug files
- [ ] Kiểm tra `detect_with_low_contrast()` (giữ nguyên)

### Phase 3: Cập nhật Main Function
- [ ] Cập nhật `detect_label_region()`: Routing logic mới
- [ ] Implement fallback chain: HIGH→LOW, LOW→STOP
- [ ] Cập nhật log output
- [ ] Cập nhật return values

### Phase 4: Cleanup
- [ ] Xóa constants không dùng
- [ ] Xóa tất cả file debug `medium_*`
- [ ] Xóa file debug `debug_02_edges.png`
- [ ] Kiểm tra imports

### Phase 5: Testing
- [ ] Test với ảnh High contrast (áo tối)
- [ ] Test với ảnh Low contrast (áo trắng)
- [ ] Verify debug visualizations
- [ ] Verify fallback chain

---

## ⚠️ Lưu Ý Quan Trọng

1. **Không sửa `main.py`**: File này chỉ gọi `detect_label_region()` và visualize, không cần thay đổi

2. **Debug files sẽ thay đổi:**
   - `debug_01_histogram.png`: Thêm đường tím cho trough
   - `debug_02_edges.png`: BỊ XÓA
   - `debug_03_contrast.png`: Không đổi
   - `debug_high_*.png`: Thay đổi text (Otsu → Simple Threshold)
   - `debug_medium_*.png`: TẤT CẢ BỊ XÓA
   - `debug_low_*.png`: Không đổi

3. **Fallback logic quan trọng:**
   - HIGH có thể fallback → LOW (vì có histogram separable)
   - LOW KHÔNG fallback → HIGH (vì histogram không separable)

4. **Trough position:**
   - Luôn nằm trong khoảng `[peak1, peak2]`
   - Nếu không tìm được 2 peaks, fallback = 127

---

## 🎯 Kết Quả Mong Đợi

### Trước khi cập nhật:
- 3 strategies: HIGH, MEDIUM, LOW
- 3 metrics: histogram, edge, contrast
- Phức tạp: Multiple fallback chains

### Sau khi cập nhật:
- **2 strategies**: HIGH, LOW (loại bỏ MEDIUM)
- **2 metrics**: histogram, contrast (loại bỏ edge)
- **Đơn giản hóa**: HIGH→LOW hoặc LOW→STOP
- **Chính xác hơn**: Dùng trough làm threshold thay vì Otsu
- **Logic rõ ràng**: `separation > 0 and score > 0.3` = High

---

## 📊 So Sánh Logic Cũ vs Mới

| Khía Cạnh | Logic Cũ | Logic Mới |
|-----------|----------|-----------|
| **Số Strategies** | 3 (HIGH/MEDIUM/LOW) | 2 (HIGH/LOW) |
| **Metrics** | 3 (histogram/edge/contrast) | 2 (histogram/contrast) |
| **Final Score** | sep×0.4 + edge×0.3 + con×0.3 | sep×0.6 + con×0.4 |
| **High Condition** | score > 0.45 | sep>0 AND score>0.3 |
| **Medium Condition** | 0.25 < score ≤ 0.45 | ❌ Loại bỏ |
| **Low Condition** | score ≤ 0.25 | Còn lại |
| **HIGH Threshold** | Otsu (auto) | Trough (optimal) |
| **Fallback HIGH** | MEDIUM→HIGH | LOW (direct) |
| **Fallback LOW** | MEDIUM→HIGH | ❌ STOP |

---

## 📂 Files Cần Sửa

### Sửa:
1. `src/label_region_extractor.py` - **TOÀN BỘ**

### Không sửa:
1. `main.py` - Giữ nguyên
2. `requirements.txt` - Giữ nguyên
3. `README.md` - Có thể cập nhật sau (optional)

### Xóa (Debug files):
1. `data/debug/debug_02_edges.png`
2. `data/debug/debug_medium_*.png` (tất cả)

---

**Tổng số thay đổi ước tính**: ~15 functions/blocks
**Độ phức tạp**: Trung bình đến Cao
**Thời gian ước tính**: 30-45 phút coding + testing

---

_Kế hoạch này đảm bảo cập nhật đầy đủ, chính xác theo yêu cầu._
_Sẵn sàng để thực hiện khi bạn xác nhận._

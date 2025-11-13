# 📋 Chiến lược phát hiện vùng nhãn - Label Region Extractor

## 🎯 Tổng quan

Hệ thống phát hiện vùng nhãn (label) trên áo công nhân dựa trên phân tích độ tương phản tự động và 3 chiến lược phát hiện khác nhau. Mỗi chiến lược được tối ưu cho một loại áo cụ thể (tối, sáng, hoặc trung bình).

---

## 🏗️ Kiến trúc hệ thống: 3 tầng

```
Ảnh đầu vào
     ↓
Preprocessing
(BGR → Grayscale → GaussianBlur 5×5)
     ↓
┌──────────────────────────────────┐
│  TẦNG 1: Phân tích tự động       │
│  - Histogram Analysis            │
│  - Edge Detection                │
│  - Contrast Measurement          │
│  → Final Score → Level           │
└──────────────────────────────────┘
     ↓
Phân loại: HIGH | MEDIUM | LOW
     ↓
┌──────────────────────────────────┐
│  TẦNG 2: Strategy Selection      │
│                                  │
│  HIGH (>0.45)                    │
│  ├─ Binary Threshold (Otsu)      │
│  └─ Morphology Operations        │
│                                  │
│  MEDIUM (0.25-0.45)              │
│  ├─ Canny Edge Detection         │
│  ├─ Strong Morphology            │
│  └─ Candidate Loop + QR          │
│                                  │
│  LOW (<0.25)                     │
│  ├─ CLAHE Enhancement            │
│  ├─ QR-First Detection           │
│  └─ Geometry Inference           │
└──────────────────────────────────┘
     ↓
Fallback Chain
(MEDIUM→HIGH, LOW→MEDIUM→HIGH)
     ↓
Kết quả: (rect, box, qr_text, qr_points, strategy_used)
```

---

## 📊 TẦNG 1: Phân tích tự động (Analysis)

### Mục đích
Phân tích ảnh để xác định mức độ tương phản và chọn chiến lược phù hợp.

### Input
- Ảnh grayscale đã qua GaussianBlur

### Output
- **Final Score**: Điểm tổng hợp (0.0 - 1.0)
- **Level**: HIGH / MEDIUM / LOW
- **Metrics**: separation, edge_strength, contrast_ratio

---

### 1. Histogram Analysis

**Mục đích:** Đo khoảng cách giữa 2 peaks chính trong histogram (độ phân tách màu sắc)

**Pseudocode:**
```
function analyze_histogram(gray_image):
    # Lấy vùng phân tích (toàn bộ ảnh)
    region = gray_image
    
    # Tính histogram 256 bins
    hist = calculate_histogram(region, bins=256)
    
    # Smooth với moving average (window=5)
    smoothed = moving_average(hist, window=5)
    
    # Tìm local maxima
    peaks = []
    threshold = mean(smoothed) * 0.5
    
    for i in range(10, 246):
        if is_local_maximum(smoothed, i) and smoothed[i] > threshold:
            if not too_close_to_existing_peaks(peaks, i, min_distance=30):
                peaks.append((position=i, height=smoothed[i]))
    
    # Chọn 2 peaks cao nhất
    if len(peaks) < 2:
        return (peak1=0, peak2=255, separation=0.0)
    
    top_2_peaks = sort_by_height(peaks)[:2]
    peak1, peak2 = sort_by_position(top_2_peaks)
    
    # Tính separation (normalized)
    separation = |peak2 - peak1| / 255.0
    
    # Debug: Vẽ histogram với peaks
    save_histogram_plot(hist, smoothed, peak1, peak2, separation)
    
    return (peak1, peak2, separation)
```

**Metrics:**
- `separation`: 0.0 - 1.0 (càng cao = hai màu phân biệt rõ)
- Ví dụ: Áo đen + nhãn trắng → separation ≈ 0.8-1.0

---

### 2. Edge Detection Analysis

**Mục đích:** Đo mật độ edges trong ảnh (độ phức tạp biên)

**Pseudocode:**
```
function analyze_edges(gray_image):
    # Lấy vùng phân tích
    region = gray_image
    
    # Canny Edge Detection
    edges = canny_edge_detection(region, 
                                  threshold1=50, 
                                  threshold2=150)
    
    # Đếm edge pixels
    edge_pixels = count_nonzero(edges)
    total_pixels = width(region) * height(region)
    
    # Tính edge strength (normalized)
    edge_strength = edge_pixels / total_pixels
    
    # Debug: Lưu ảnh edges
    save_edge_comparison(region, edges, edge_strength, edge_pixels)
    
    return (edge_pixels, edge_strength)
```

**Metrics:**
- `edge_strength`: 0.0 - 1.0 (càng cao = nhiều biên)
- Thường: 0.01 - 0.05 (1-5% pixels là edges)

---

### 3. Contrast Measurement

**Mục đích:** Đo độ chênh lệch cường độ sáng (standard deviation)

**Pseudocode:**
```
function analyze_contrast(gray_image):
    # Lấy vùng phân tích
    region = gray_image
    
    # Tính mean và standard deviation
    mean_intensity = mean(region)
    stddev_intensity = standard_deviation(region)
    
    # Normalize contrast ratio
    contrast_ratio = stddev_intensity / 128.0
    
    # Debug: Lưu JSON và histogram intensity
    data = {
        "mean": mean_intensity,
        "stddev": stddev_intensity,
        "contrast_ratio": contrast_ratio,
        "min": min(region),
        "max": max(region)
    }
    
    save_json(data, "contrast_metrics.json")
    save_intensity_histogram(region, mean_intensity, stddev_intensity)
    
    return (mean_intensity, stddev_intensity, contrast_ratio)
```

**Metrics:**
- `contrast_ratio`: 0.0 - 2.0 (thường 0.1 - 0.5)
- Cao = nhiều biến đổi cường độ

---

### 4. Final Score Calculation

**Pseudocode:**
```
function analyze_frame(gray_image):
    # Gọi 3 hàm phân tích
    (peak1, peak2, separation) = analyze_histogram(gray_image)
    (edge_pixels, edge_strength) = analyze_edges(gray_image)
    (mean, stddev, contrast_ratio) = analyze_contrast(gray_image)
    
    # Normalize edge_strength
    EDGE_MAX = 0.1
    edge_strength_norm = min(edge_strength / EDGE_MAX, 1.0)
    
    # Weighted sum
    final_score = separation * 0.4 +           # 40% weight
                  edge_strength_norm * 0.3 +   # 30% weight
                  contrast_ratio * 0.3         # 30% weight
    
    # Phân loại level
    if final_score > 0.45:
        level = "HIGH"
    else if final_score > 0.25:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    return {
        level: level,
        final_score: final_score,
        separation: separation,
        edge_strength: edge_strength,
        contrast_ratio: contrast_ratio,
        # ... debug info
    }
```

**Thresholds:**
- `HIGH_THRESHOLD = 0.45` (áo đen/đậm)
- `MEDIUM_THRESHOLD = 0.25` (áo màu nhạt)
- `< 0.25` = LOW (áo trắng/kem)

---

## 🎯 TẦNG 2: Chiến lược phát hiện (Detection Strategies)

### Strategy HIGH: Binary Threshold + Morphology

**Áp dụng cho:** Áo tối/màu đậm (đen, xanh đậm, đỏ đậm...)

**Đặc điểm:**
- Separation cao (ảnh có 2 màu phân biệt rõ)
- Edge strength cao
- Dễ tách nhãn bằng threshold

**Pseudocode:**
```
function detect_with_high_contrast(image, gray):
    # Otsu adaptive threshold (tự động tìm ngưỡng tối ưu)
    threshold_value, binary = otsu_threshold(gray)
    
    # Morphology operations
    kernel_3x3 = rectangular_kernel(3, 3)
    
    # Open: Loại bỏ noise nhỏ
    morph = morphology_open(binary, kernel_3x3, iterations=1)
    
    # Close: Lấp lỗ hổng
    morph = morphology_close(morph, kernel_3x3, iterations=2)
    
    # Tìm contours
    contours = find_contours(morph, mode=EXTERNAL)
    
    if no_contours_found:
        return FAIL
    
    # Chọn contour lớn nhất
    largest_contour = max(contours, key=contour_area)
    
    # Tính rotated rectangle
    rect = min_area_rect(largest_contour)
    box = get_4_corners(rect)
    
    # Crop vùng và verify QR code
    bounding_rect = get_bounding_rect(largest_contour)
    label_roi = crop_image(image, bounding_rect)
    
    qr_text, qr_points_local = detect_qr_code(label_roi)
    
    if qr_text found:
        # Convert QR points to global coordinates
        qr_points_global = qr_points_local + bounding_rect.top_left
        return (rect, box, qr_text, qr_points_local, qr_points_global)
    
    return FAIL
```

**Tham số:**
- Otsu threshold: Tự động
- Morphology kernel: 3×3
- Open iterations: 1
- Close iterations: 2

---

### Strategy MEDIUM: Canny Edge + Strong Morphology

**Áp dụng cho:** Áo màu trung bình (xám, xanh nhạt, vàng...)

**Đặc điểm:**
- Separation trung bình
- Cần edge detection để tách biên
- Nhiều candidate cần verify QR

**Pseudocode:**
```
function detect_with_medium_contrast(image, gray):
    # Canny với threshold thấp (nhạy hơn)
    edges = canny_edge_detection(gray, 
                                  threshold1=30,  # Lower
                                  threshold2=100) # Lower
    
    # Strong morphology (kernel lớn)
    kernel_7x7 = rectangular_kernel(7, 7)
    
    # Close: Nối các edges gần nhau
    edges = morphology_close(edges, kernel_7x7, iterations=3)
    
    # Dilate: Làm dày edges
    edges = morphology_dilate(edges, kernel_7x7, iterations=1)
    
    # Tìm contours
    contours = find_contours(edges, mode=EXTERNAL)
    
    if no_contours_found:
        return FAIL
    
    # Filter theo area ratio
    image_area = width(image) * height(image)
    candidates = []
    
    for contour in contours:
        area = contour_area(contour)
        area_ratio = area / image_area
        
        # Chỉ giữ contours có kích thước hợp lý
        if 0.05 <= area_ratio <= 0.80:  # 5%-80%
            rect = min_area_rect(contour)
            candidates.append((contour, rect, area))
    
    # Sort theo area (lớn nhất trước)
    candidates = sort_by_area(candidates, descending=True)
    
    # Loop qua candidates và verify QR (early exit)
    for (contour, rect, area) in candidates:
        bounding_rect = get_bounding_rect(contour)
        label_roi = crop_image(image, bounding_rect)
        
        qr_text, qr_points_local = detect_qr_code(label_roi)
        
        if qr_text found:
            box = get_4_corners(rect)
            qr_points_global = qr_points_local + bounding_rect.top_left
            return (rect, box, qr_text, qr_points_local, qr_points_global)
    
    return FAIL
```

**Tham số:**
- Canny: 30/100 (thấp hơn HIGH)
- Morphology kernel: 7×7 (lớn hơn HIGH)
- Close iterations: 3
- Dilate iterations: 1
- Area filter: 5%-80%

---

### Strategy LOW: QR-First + Geometry Inference

**Áp dụng cho:** Áo sáng/trắng/kem (nhãn gần như không có viền rõ)

**Đặc điểm:**
- Separation thấp (màu nhãn gần với màu áo)
- Edge detection không hiệu quả
- **Chiến lược:** Tìm QR code trước, suy luận vị trí nhãn

**Pseudocode:**
```
function detect_with_low_contrast(image, gray):
    # CLAHE preprocessing (đã apply trước khi gọi hàm)
    # → Tăng contrast cục bộ
    
    # Histogram equalization cho QR detection
    enhanced = histogram_equalization(gray)
    
    # Detect QR code (try enhanced first, fallback to original)
    qr_text, qr_points = detect_qr_code(enhanced)
    
    if not qr_text:
        qr_text, qr_points = detect_qr_code(image)
    
    if not qr_text or len(qr_points) < 4:
        return FAIL
    
    # Tính geometry QR code
    p0 = qr_points[0]  # top-left
    p1 = qr_points[1]  # top-right
    p3 = qr_points[3]  # bottom-left
    
    # Vectors
    top_vec = p1 - p0
    left_vec = p3 - p0
    
    # QR dimensions
    qr_width = length(top_vec)
    qr_height = length(left_vec)
    
    # Unit vectors
    dir_right = normalize(top_vec)
    dir_down = normalize(left_vec)
    
    # Góc xoay
    angle = arctan2(top_vec.y, top_vec.x) * 180 / π
    
    # Suy luận label dimensions
    LABEL_WIDTH_RATIO = 4.0   # Label rộng = 4 × QR
    LABEL_HEIGHT_RATIO = 3.0  # Label cao = 3 × QR
    
    label_width = qr_width * LABEL_WIDTH_RATIO
    label_height = qr_height * LABEL_HEIGHT_RATIO
    
    # Construct 4 corners
    # Giả định: QR ở TRÁI DƯỚI của label
    # → Expand PHẢI và LÊN TRÊN
    
    qr_top_left = p0
    
    # Label top-left: Đi lên từ QR top-left
    label_top_left = qr_top_left - dir_down * (label_height - qr_height)
    
    # Label corners
    label_top_right = label_top_left + dir_right * label_width
    label_bottom_left = label_top_left + dir_down * label_height
    label_bottom_right = label_bottom_left + dir_right * label_width
    
    # Tạo RotatedRect
    label_center = (label_top_left + label_top_right + 
                    label_bottom_right + label_bottom_left) / 4
    
    rect = RotatedRect(center=label_center, 
                       size=(label_width, label_height), 
                       angle=angle)
    
    box = [label_top_left, label_top_right, 
           label_bottom_right, label_bottom_left]
    
    # Note: qr_points_180 = None (detect trên toàn ảnh)
    return (rect, box, qr_text, None, qr_points)
```

**Tham số:**
- CLAHE: clipLimit=2.0, tileGridSize=8×8
- Histogram equalization: Full range
- Label expansion: 4.0× width, 3.0× height
- QR position: Left-bottom của label

**Geometry Inference:**
```
┌─────────────────────────────────┐  ← Label top-left
│                                 │
│                                 │  ← Label height = 3 × QR height
│  ┌────┐                        │
│  │ QR │ ← QR code (left-bottom)│
│  └────┘                        │
└─────────────────────────────────┘
 ← Label width = 4 × QR width
```

---

## 🔄 Fallback Chain (Chuỗi dự phòng)

### Chiến lược
Nếu strategy chính thất bại → Thử strategy mạnh hơn

```
HIGH: Không có fallback (đã mạnh nhất)

MEDIUM: → HIGH
  ├─ Canny fail → Thử Binary Threshold
  └─ Lý do: HIGH robust hơn với contour lớn

LOW: → MEDIUM → HIGH
  ├─ QR không detect → Thử Canny
  ├─ Canny fail → Thử Binary
  └─ Lý do: LOW phụ thuộc QR, dễ fail
```

### Pseudocode
```
function detect_label_region(image):
    # Preprocessing
    gray = convert_to_grayscale(image)
    gray = gaussian_blur(gray, kernel_size=5)
    
    # TẦNG 1: Phân tích
    analysis = analyze_frame(gray)
    
    log_analysis_metrics(analysis)
    
    # TẦNG 2: Routing
    result = None
    strategy_used = ""
    
    if analysis.level == "HIGH":
        result = detect_with_high_contrast(image, gray)
        if result:
            strategy_used = "HIGH"
    
    else if analysis.level == "MEDIUM":
        result = detect_with_medium_contrast(image, gray)
        if result:
            strategy_used = "MEDIUM"
        else:
            # Fallback to HIGH
            result = detect_with_high_contrast(image, gray)
            if result:
                strategy_used = "MEDIUM→HIGH"
    
    else if analysis.level == "LOW":
        # Apply CLAHE preprocessing
        gray = apply_clahe(gray, clip_limit=2.0, tile_size=8)
        
        result = detect_with_low_contrast(image, gray)
        if result:
            strategy_used = "LOW"
        else:
            # Fallback to MEDIUM
            result = detect_with_medium_contrast(image, gray)
            if result:
                strategy_used = "LOW→MEDIUM"
            else:
                # Fallback to HIGH
                result = detect_with_high_contrast(image, gray)
                if result:
                    strategy_used = "LOW→MEDIUM→HIGH"
    
    if result:
        log_success(result, strategy_used)
        return (result..., strategy_used)
    else:
        log_failure()
        return (None, None, None, None, None, None)
```

---

## 🔍 Debug và Visualization

### TẦNG 1 - Analysis Debug

**1. Histogram Analysis**
```
Output: debug_01_histogram.png
- Subplot 1: Histogram plot
  • Original histogram (gray, alpha=0.5)
  • Smoothed histogram (blue)
  • Peak 1 (red vertical line)
  • Peak 2 (green vertical line)
  • Separation score in title
- Subplot 2: Analysis region (grayscale image)
```

**2. Edge Detection**
```
Output: debug_02_edges.png
- Subplot 1: Original image
- Subplot 2: Canny edges
  • Edge strength + pixel count in title
```

**3. Contrast Measurement**
```
Output: debug_03_contrast.json
{
  "mean_intensity": 123.45,
  "stddev_intensity": 30.12,
  "contrast_ratio": 0.235,
  "min_intensity": 0,
  "max_intensity": 255,
  "image_shape": [586, 958]
}

Output: debug_03_contrast.png
- Subplot 1: Analysis region with colorbar
- Subplot 2: Intensity distribution histogram
  • Mean line (red)
  • Mean±σ lines (orange)
```

### TẦNG 2 - Strategy Debug

Mỗi strategy in log chi tiết:
```
Strategy HIGH:
  → Otsu threshold: 171.0
  → Found 4 contours
  → Largest area: 537338 pixels
  ✓ QR detected: "111625-TX-M-005540-2"

Strategy MEDIUM:
  → Lower Canny thresholds (30/100)
  → Found 12 contours
  → 3 candidates after filtering (area 5-80%)
  ✓ QR found in candidate #2 (area=45230)

Strategy LOW:
  → Applied CLAHE preprocessing
  → Applied histogram equalization
  ✓ QR detected: "111625-TX-M-005540-2"
  → QR geometry: 123.4×120.1 px, angle=5.2°
  → Predicted label: 493.6×360.3 px
```

---

## 📐 Constants và Thresholds

### Analysis Thresholds
```
HIGH_THRESHOLD = 0.45       # Final score > 0.45 → HIGH strategy
MEDIUM_THRESHOLD = 0.25     # Final score > 0.25 → MEDIUM strategy
EDGE_MAX = 0.1              # Normalization cap for edge_strength
```

### Label Geometry (Strategy LOW)
```
LABEL_WIDTH_RATIO = 4.0     # Label width = 4 × QR width
LABEL_HEIGHT_RATIO = 3.0    # Label height = 3 × QR height
```

### Morphology Parameters
```
HIGH Strategy:
  - Kernel: 3×3
  - Open iterations: 1
  - Close iterations: 2

MEDIUM Strategy:
  - Kernel: 7×7
  - Close iterations: 3
  - Dilate iterations: 1

LOW Strategy:
  - CLAHE clip limit: 2.0
  - CLAHE tile size: 8×8
```

### Edge Detection
```
HIGH Strategy (center analysis):
  - Canny: 50/150

MEDIUM Strategy (full detection):
  - Canny: 30/100 (lower = more sensitive)
```

### Area Filtering (MEDIUM Strategy)
```
Min area ratio: 5%  (0.05)
Max area ratio: 80% (0.80)
```

---

## 🧪 Test Cases

### Test Case 1: Áo đen + nhãn trắng
```
Expected:
  - Analysis: HIGH (score ≈ 0.6-0.8)
  - Strategy: HIGH
  - Result: SUCCESS (Binary threshold dễ tách)
```

### Test Case 2: Áo xám + nhãn trắng
```
Expected:
  - Analysis: MEDIUM (score ≈ 0.3-0.5)
  - Strategy: MEDIUM
  - Result: SUCCESS (Canny detect edges)
  - Fallback: Có thể cần HIGH nếu contrast thấp hơn dự kiến
```

### Test Case 3: Áo trắng/kem + nhãn trắng
```
Expected:
  - Analysis: LOW (score < 0.25)
  - Strategy: LOW → MEDIUM → HIGH (fallback chain)
  - Result: Tùy thuộc QR detection
  - Note: QR phải rõ nét để LOW thành công
```

### Test Case 4: Ảnh guide-box (không có QR)
```
Expected:
  - Analysis: LOW (score ≈ 0.05-0.15)
  - Strategy: LOW fail (no QR) → MEDIUM fail → HIGH fail
  - Result: FAIL (không có QR để verify)
```

---

## 📝 Lưu ý triển khai

### 1. Vùng phân tích
**Trước đây:** Lấy 1/3 center region (cho ảnh lớn, giả định nhãn ở giữa)
**Hiện tại:** Lấy toàn bộ ảnh (vì input đã là vùng nhỏ chứa nhãn)

```
# Helper function dùng chung
function get_analysis_region(gray_image):
    return gray_image  # Toàn bộ ảnh
```

### 2. QR Code Points
- Detect trên ROI cục bộ → `qr_points_180` (coordinates trong ROI)
- Convert sang global coordinates → `qr_points` (coordinates trong ảnh gốc)
- Strategy LOW detect trên full image → `qr_points_180 = None`

### 3. RotatedRect
```
RotatedRect structure:
  - center: (x, y)
  - size: (width, height)
  - angle: degrees (0-360)
  
Box points (4 corners):
  [top_left, top_right, bottom_right, bottom_left]
```

### 4. Coordinate System
- Image indexing: `image[y, x]` (row, col)
- Point format: `(x, y)` hoặc `[x, y]`
- QR points order: `[top_left, top_right, bottom_right, bottom_left]`

### 5. Fallback Priority
```
Mức độ robust (cao → thấp):
  HIGH > MEDIUM > LOW

Lý do:
  - HIGH: Binary threshold + morphology → Ổn định với contour rõ
  - MEDIUM: Canny edges → Nhạy với noise nhưng linh hoạt
  - LOW: Phụ thuộc QR detection → Dễ fail nếu QR mờ/nghiêng
```

---

## 🚀 Flow tổng quát

```
1. Load image
     ↓
2. Preprocessing (BGR → Gray → Blur)
     ↓
3. TẦNG 1: Analysis
   ├─ analyze_histogram() → separation
   ├─ analyze_edges() → edge_strength
   ├─ analyze_contrast() → contrast_ratio
   └─ analyze_frame() → final_score + level
     ↓
4. Log metrics và level
     ↓
5. TẦNG 2: Strategy selection
   ├─ if HIGH: detect_with_high_contrast()
   ├─ if MEDIUM: detect_with_medium_contrast() [→ HIGH]
   └─ if LOW: detect_with_low_contrast() [→ MEDIUM → HIGH]
     ↓
6. Return result + strategy_used
     ↓
7. Visualization
   ├─ Draw label box (green)
   ├─ Draw QR points (red)
   └─ Display info (strategy, QR text)
     ↓
8. Save output
```

---

## 🎯 Key Takeaways

### Điểm mạnh của hệ thống
1. **Tự động phân loại:** Không cần biết trước màu áo
2. **Robust fallback:** Nhiều tầng dự phòng
3. **Tối ưu từng loại:** Mỗi strategy cho 1 use case cụ thể
4. **Debug-friendly:** Visualization đầy đủ cho mỗi bước

### Điểm yếu
1. **Phụ thuộc QR:** LOW strategy hoàn toàn dựa vào QR detection
2. **Tham số cố định:** LABEL_WIDTH_RATIO, LABEL_HEIGHT_RATIO có thể không đúng với mọi label
3. **Không xử lý:** Ảnh bị mờ, nghiêng nhiều, hoặc nhãn bị che khuất

### Cải thiện có thể
1. **Adaptive ratios:** Học width/height ratio từ dữ liệu thực tế
2. **Deep learning:** CNN cho label segmentation
3. **Better QR detector:** Dùng pyzbar hoặc zxing-cpp thay cv2.QRCodeDetector
4. **Perspective correction:** Xử lý ảnh bị nghiêng trước khi detect

---

**Version:** 2.0 (Actual Implementation)  
**Last Update:** November 13, 2025  
**Status:** Deployed and tested

# 📋 Kế hoạch triển khai Python - Label Region Extractor

## 🎯 Mục tiêu

Chuyển đổi logic `LabelRegionExtractor.cs` sang Python để test/debug trước khi áp dụng vào production C#. Triển khai **chính xác 100%** logic từ mã nguồn C# thực tế (không theo kế hoạch cũ vì có sự khác biệt).

---

## 📊 Kiến trúc tổng thể: Hệ thống 3 tầng (Theo mã C# thực tế)

```
Ảnh đầu vào (src)
         ↓
    Preprocessing
    (BGR → Gray → GaussianBlur 5×5)
         ↓
┌─────────────────────────────────┐
│  TẦNG 1: Phân tích tự động      │
│  3 Metrics → Final Score        │
│  → ContrastLevel                │
└─────────────────────────────────┘
         ↓
    Phân loại theo Level
         ↓
   ┌─────┴──────┬──────────┐
   ↓            ↓          ↓
HIGH         MEDIUM       LOW
(>0.45)    (0.25-0.45)  (<0.25)
   ↓            ↓          ↓
┌────────┐  ┌────────┐  ┌────────┐
│Binary  │  │Canny   │  │CLAHE + │
│Otsu    │  │30/100  │  │QR-First│
│+Morph  │  │+Loop   │  │Geometry│
└────────┘  └────────┘  └────────┘
   ↓            ↓          ↓
   └─────┬──────┴──────────┘
         ↓
     Fallback Chain
  (MEDIUM→HIGH, LOW→MEDIUM→HIGH)
         ↓
  (rect, box, qrText, qrPoints180, qrPoints, strategyUsed)
```

---

## 🔧 Constants (Theo mã C# thực tế)

```python
# Analysis thresholds
HIGH_THRESHOLD = 0.45      # Nới rộng từ 0.6 → 0.45
MEDIUM_THRESHOLD = 0.25    # Thu hẹp từ 0.3 → 0.25
EDGE_MAX = 0.1             # Normalization cho edge strength

# Label expansion ratios (cho Strategy LOW)
LABEL_WIDTH_RATIO = 4.0    # Giảm từ 9.0 → 4.0
LABEL_HEIGHT_RATIO = 3.0   # Giữ nguyên 3×
```

**Lưu ý:** Mã C# đã được điều chỉnh khác so với kế hoạch ban đầu!

---

## 📌 TẦNG 1: Analysis Methods

### 1. `analyze_histogram(gray: np.ndarray) -> tuple[int, int, float]`

**Input:** Ảnh grayscale (height, width)  
**Output:** `(peak1_pos, peak2_pos, separation)`

**Chi tiết implementation:**

```python
def analyze_histogram(gray):
    """
    Phân tích histogram để tìm 2 peaks chính và tính separation.
    
    Logic từ C#:
    1. Lấy vùng center (1/3 kích thước nhỏ nhất)
    2. Tính histogram 256 bins
    3. Smooth bằng moving average 5 bins
    4. Tìm local maxima (> 0.5×avgHeight, cách nhau >30 bins)
    5. Chọn 2 peaks cao nhất, sort theo position
    6. separation = |peak2 - peak1| / 255.0
    """
    
    # 1. Center region (1/3 min dimension)
    sample_size = min(gray.shape[1], gray.shape[0]) // 3
    cx = gray.shape[1] // 2
    cy = gray.shape[0] // 2
    x1 = cx - sample_size // 2
    y1 = cy - sample_size // 2
    center_roi = gray[y1:y1+sample_size, x1:x1+sample_size]
    
    # 2. Histogram
    hist = cv2.calcHist([center_roi], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # 3. Smooth (moving average 5 bins)
    smoothed = np.copy(hist)
    for i in range(2, 254):
        smoothed[i] = np.mean(hist[i-2:i+3])
    
    # 4. Find local maxima
    avg_height = np.mean(smoothed)
    threshold = avg_height * 0.5
    
    peaks = []
    for i in range(10, 246):
        is_local_max = (smoothed[i] > smoothed[i-1] and 
                       smoothed[i] > smoothed[i+1] and 
                       smoothed[i] > threshold)
        
        if is_local_max:
            # Check not too close to existing peaks
            too_close = any(abs(p[0] - i) < 30 for p in peaks)
            if not too_close:
                peaks.append((i, smoothed[i]))
    
    # 5. Take 2 highest peaks
    if len(peaks) < 2:
        return (0, 255, 0.0)
    
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
    peaks = sorted(peaks, key=lambda x: x[0])  # Sort by position
    
    peak1 = peaks[0][0]
    peak2 = peaks[1][0]
    separation = abs(peak2 - peak1) / 255.0
    
    return (peak1, peak2, separation)
```

---

### 2. `analyze_edges(gray: np.ndarray) -> tuple[int, float]`

**Input:** Ảnh grayscale  
**Output:** `(edge_pixels, edge_strength)`

```python
def analyze_edges(gray):
    """
    Phân tích edges bằng Canny để tính edge strength.
    
    Logic từ C#:
    1. Lấy vùng center (1/3 kích thước)
    2. Canny(50, 150)
    3. Đếm non-zero pixels
    4. edge_strength = edge_pixels / total_pixels
    """
    
    # 1. Center region
    sample_size = min(gray.shape[1], gray.shape[0]) // 3
    cx = gray.shape[1] // 2
    cy = gray.shape[0] // 2
    x1 = cx - sample_size // 2
    y1 = cy - sample_size // 2
    center_roi = gray[y1:y1+sample_size, x1:x1+sample_size]
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(center_roi, threshold1=50, threshold2=150)
    
    # 3. Count edge pixels
    edge_pixels = cv2.countNonZero(edges)
    total_pixels = center_roi.shape[0] * center_roi.shape[1]
    
    edge_strength = edge_pixels / total_pixels
    
    return (edge_pixels, edge_strength)
```

---

### 3. `analyze_contrast(gray: np.ndarray) -> tuple[float, float, float]`

**Input:** Ảnh grayscale  
**Output:** `(mean, stddev, contrast_ratio)`

```python
def analyze_contrast(gray):
    """
    Phân tích contrast bằng standard deviation.
    
    Logic từ C#:
    1. Lấy vùng center (1/3 kích thước)
    2. Tính mean, stddev
    3. contrast_ratio = stddev / 128.0
    """
    
    # 1. Center region
    sample_size = min(gray.shape[1], gray.shape[0]) // 3
    cx = gray.shape[1] // 2
    cy = gray.shape[0] // 2
    x1 = cx - sample_size // 2
    y1 = cy - sample_size // 2
    center_roi = gray[y1:y1+sample_size, x1:x1+sample_size]
    
    # 2. Calculate mean and stddev
    mean, stddev = cv2.meanStdDev(center_roi)
    mean = mean[0][0]
    stddev = stddev[0][0]
    
    # 3. Contrast ratio
    contrast_ratio = stddev / 128.0
    
    return (mean, stddev, contrast_ratio)
```

---

### 4. `analyze_frame(gray: np.ndarray) -> ContrastAnalysisResult`

**Input:** Ảnh grayscale  
**Output:** Dataclass chứa level + metrics

```python
@dataclass
class ContrastAnalysisResult:
    """Kết quả phân tích độ tương phản."""
    level: str  # 'High', 'Medium', 'Low'
    final_score: float
    
    # 3 metrics
    separation: float
    edge_strength: float
    contrast_ratio: float
    
    # Debug info
    peak1_position: int
    peak2_position: int
    edge_pixel_count: int
    mean_intensity: float
    stddev_intensity: float


def analyze_frame(gray):
    """
    Phân tích frame để tính final score và xác định contrast level.
    
    Logic từ C#:
    1. Gọi 3 hàm phân tích
    2. Normalize edge_strength (min(edge_strength / 0.1, 1.0))
    3. Final Score = separation×0.4 + edge_strength_norm×0.3 + contrast_ratio×0.3
    4. Phân loại: >0.45=High, 0.25-0.45=Medium, <0.25=Low
    """
    
    # 1. Call 3 analysis methods
    peak1, peak2, separation = analyze_histogram(gray)
    edge_pixels, edge_strength = analyze_edges(gray)
    mean, stddev, contrast_ratio = analyze_contrast(gray)
    
    # 2. Normalize edge strength
    edge_strength_norm = min(edge_strength / EDGE_MAX, 1.0)
    
    # 3. Calculate Final Score
    final_score = (separation * 0.4 + 
                   edge_strength_norm * 0.3 + 
                   contrast_ratio * 0.3)
    
    # 4. Determine level
    if final_score > HIGH_THRESHOLD:
        level = 'High'
    elif final_score > MEDIUM_THRESHOLD:
        level = 'Medium'
    else:
        level = 'Low'
    
    # 5. Return result
    return ContrastAnalysisResult(
        level=level,
        final_score=final_score,
        separation=separation,
        edge_strength=edge_strength,
        contrast_ratio=contrast_ratio,
        peak1_position=peak1,
        peak2_position=peak2,
        edge_pixel_count=edge_pixels,
        mean_intensity=mean,
        stddev_intensity=stddev
    )
```

---

## 📌 TẦNG 2: Strategy Methods

### Strategy HIGH: Binary Threshold + Morphology

```python
def detect_with_high_contrast(src, gray, threshold_value=150):
    """
    Strategy HIGH: Binary Threshold + Morphology (cho áo tối/màu đậm).
    
    Logic từ C# (đã cập nhật):
    1. Otsu adaptive threshold thay vì hardcoded value
    2. Morphology: Open(3×3, 1 iter) → Close(3×3, 2 iters)
    3. FindContours(EXTERNAL)
    4. Chọn largest contour theo area
    5. MinAreaRect → box
    6. Crop bounding rect → QR verification
    7. Trả về (rect, box, qr_text, qr_points_180, qr_points) hoặc (None, None, None, None, None)
    
    Returns:
        tuple: (RotatedRect dict, box points, qr_text, qr_points_180, qr_points)
    """
    
    print("  → Method: Binary Threshold + Morphology")
    
    # 1. Otsu adaptive threshold
    _, binary = cv2.threshold(gray, 0, 255, 
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu_threshold = _  # OpenCV returns threshold value
    print(f"  → Otsu adaptive threshold: {otsu_threshold:.1f}")
    
    # 2. Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("  ✗ No contours found")
        return (None, None, None, None, None)
    
    print(f"  → Found {len(contours)} contours")
    
    # 4. Find largest contour
    biggest = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(biggest)
    print(f"  → Largest contour area: {max_area:.0f} pixels")
    
    # 5. MinAreaRect
    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 6. Crop and verify QR
    bound = cv2.boundingRect(biggest)
    x, y, w, h = bound
    
    # Clamp to image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(src.shape[1] - x, w)
    h = min(src.shape[0] - y, h)
    
    label_roi = src[y:y+h, x:x+w]
    
    # QR detection
    qr_detector = cv2.QRCodeDetector()
    qr_text, qr_points_180, _ = qr_detector.detectAndDecode(label_roi)
    
    qr_points = None
    if qr_points_180 is not None:
        # Convert to global coordinates
        qr_points = qr_points_180.copy()
        qr_points[:, 0] += x
        qr_points[:, 1] += y
    
    if qr_text:
        print(f"  ✓ QR detected: {qr_text}")
        return (rect, box, qr_text, qr_points_180, qr_points)
    
    print("  ✗ No QR code found in label region")
    return (None, None, None, None, None)
```

---

### Strategy MEDIUM: Canny Edge Detection

```python
def detect_with_medium_contrast(src, gray):
    """
    Strategy MEDIUM: Canny Edge + Strong Morphology (cho áo màu nhạt).
    
    Logic từ C# (đã cập nhật):
    1. Canny(30, 100) - Lower thresholds để nhạy hơn
    2. Morphology: Close(7×7, 3 iters) → Dilate(7×7, 1 iter)
    3. FindContours(EXTERNAL)
    4. Filter CHỈ theo area ratio (5-80%)
    5. Sort theo area (lớn nhất trước)
    6. Loop candidates → verify QR → Early exit khi tìm thấy
    
    Returns:
        tuple: (RotatedRect dict, box points, qr_text, qr_points_180, qr_points)
    """
    
    print("  → Method: Canny Edge + Strong Morphology")
    
    # 1. Canny with lower thresholds
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    print("  → Lower Canny thresholds (30/100) for better edge detection")
    
    # 2. Strong morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 3. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("  ✗ No contours found")
        return (None, None, None, None, None)
    
    print(f"  → Found {len(contours)} contours")
    
    # 4. Filter by area ratio (5-80%)
    roi_area = gray.shape[0] * gray.shape[1]
    candidates = []
    
    for c in contours:
        area = cv2.contourArea(c)
        area_ratio = area / roi_area
        
        if 0.05 <= area_ratio <= 0.80:
            rect = cv2.minAreaRect(c)
            candidates.append((c, rect, area))
    
    # 5. Sort by area (largest first)
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    print(f"  → {len(candidates)} candidates after filtering (area 5-80%)")
    
    # 6. Loop and verify QR (Early Exit)
    for contour, rect, area in candidates:
        bound = cv2.boundingRect(contour)
        x, y, w, h = bound
        
        # Clamp to bounds
        x = max(0, x)
        y = max(0, y)
        w = min(src.shape[1] - x, w)
        h = min(src.shape[0] - y, h)
        
        if w <= 0 or h <= 0:
            continue
        
        label_roi = src[y:y+h, x:x+w]
        
        # QR detection
        qr_detector = cv2.QRCodeDetector()
        qr_text, qr_points_180, _ = qr_detector.detectAndDecode(label_roi)
        
        qr_points = None
        if qr_points_180 is not None:
            qr_points = qr_points_180.copy()
            qr_points[:, 0] += x
            qr_points[:, 1] += y
        
        if qr_text:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            print(f"  ✓ QR detected in candidate (area={area:.0f}): {qr_text}")
            return (rect, box, qr_text, qr_points_180, qr_points)
    
    print("  ✗ No QR found in any candidate")
    return (None, None, None, None, None)
```

---

### Strategy LOW: QR-First Geometry

```python
def detect_with_low_contrast(src, gray):
    """
    Strategy LOW: QR-First + Geometry Inference (cho áo trắng/kem).
    
    Logic từ C# (đã cập nhật):
    1. Apply CLAHE preprocessing để enhance contrast cục bộ
    2. Histogram equalization cho QR detection robustness
    3. Detect QR trên enhanced image (fallback to original nếu fail)
    4. Tính geometry QR (vectors, width, height, angle)
    5. Suy luận label với expansion ratios (4.0×, 3.0×)
    6. Construct 4 corners (QR ở TRÁI DƯỚI, expand PHẢI + TRÊN)
    7. Tạo RotatedRect từ 4 corners
    
    Returns:
        tuple: (RotatedRect dict, box points, qr_text, None, qr_points)
        Note: qr_points_180 = None vì detect trên toàn ảnh
    """
    
    print("  → Method: QR-First + Geometry Inference")
    
    # 1. CLAHE preprocessing (đã được apply trước khi gọi hàm này trong C#)
    # Note: Trong C# CLAHE được apply ở switch-case, không trong hàm
    # Nhưng để đảm bảo tính đồng nhất, ta áp dụng lại ở đây
    
    # 2. Enhance contrast cho QR detection
    enhanced = cv2.equalizeHist(gray)
    print("  → Applied histogram equalization for QR detection robustness")
    
    # 3. Detect QR
    qr_detector = cv2.QRCodeDetector()
    
    # Try on enhanced first
    qr_text, qr_points, _ = qr_detector.detectAndDecode(enhanced)
    
    # Fallback to original
    if not qr_text:
        qr_text, qr_points, _ = qr_detector.detectAndDecode(src)
    
    if not qr_text or qr_points is None or len(qr_points) < 4:
        print("  ✗ No QR code detected")
        return (None, None, None, None, None)
    
    print(f"  ✓ QR detected: {qr_text}")
    
    # 4. Calculate QR geometry
    p0 = qr_points[0]  # top-left
    p1 = qr_points[1]  # top-right
    p3 = qr_points[3]  # bottom-left
    
    top_vec = p1 - p0
    left_vec = p3 - p0
    
    qr_width = np.linalg.norm(top_vec)
    qr_height = np.linalg.norm(left_vec)
    
    # Unit vectors
    dir_right = top_vec / qr_width
    dir_down = left_vec / qr_height
    dir_left = -dir_right
    
    angle_rad = np.arctan2(top_vec[1], top_vec[0])
    angle_deg = angle_rad * 180.0 / np.pi
    print(f"  → QR geometry: {qr_width:.1f}×{qr_height:.1f} px, angle={angle_deg:.1f}°")
    
    # 5. Infer label dimensions
    label_width = qr_width * LABEL_WIDTH_RATIO
    label_height = qr_height * LABEL_HEIGHT_RATIO
    print(f"  → Predicted label: {label_width:.1f}×{label_height:.1f} px")
    print(f"  → Expansion: width={LABEL_WIDTH_RATIO}×QR, height={LABEL_HEIGHT_RATIO}×QR")
    
    # 6. Calculate 4 corners
    # QR ở TRÁI DƯỚI của label → expand PHẢI và LÊN TRÊN
    qr_top_left = p0
    
    # Label top-left: đi lên trên từ QR top-left
    label_top_left = qr_top_left - dir_down * (label_height - qr_height)
    
    # Label top-right: từ top-left đi sang phải
    label_top_right = label_top_left + dir_right * label_width
    
    # Label bottom-left: từ top-left đi xuống
    label_bottom_left = label_top_left + dir_down * label_height
    
    # Label bottom-right: từ bottom-left đi sang phải
    label_bottom_right = label_bottom_left + dir_right * label_width
    
    # 7. Create RotatedRect
    label_center = (label_top_left + label_top_right + 
                   label_bottom_right + label_bottom_left) / 4.0
    
    angle = angle_deg
    
    # RotatedRect as dict (Python doesn't have C# RotatedRect)
    rect = {
        'center': tuple(label_center),
        'size': (label_width, label_height),
        'angle': angle
    }
    
    box = np.array([label_top_left, label_top_right, 
                    label_bottom_right, label_bottom_left], dtype=np.int32)
    
    print(f"  ✓ Label constructed: center=({label_center[0]:.1f},{label_center[1]:.1f}), angle={angle:.1f}°")
    
    # qr_points_180 = None (detect trên toàn ảnh, không có ROI cục bộ)
    return (rect, box, qr_text, None, qr_points)
```

---

## 📌 Hàm chính: detect_label_region()

```python
def detect_label_region(src, threshold_value=150):
    """
    Phát hiện vùng nhãn trong ảnh.
    
    Logic từ C# (flow chính xác):
    1. Preprocessing: BGR → Gray → GaussianBlur(5×5)
    2. TẦNG 1: Phân tích (AnalyzeFrame) → ContrastAnalysisResult
    3. Log analysis metrics
    4. TẦNG 2: Routing theo level:
       - HIGH: DetectWithHighContrast()
       - MEDIUM: DetectWithMediumContrast() → fallback to HIGH
       - LOW: Apply CLAHE → DetectWithLowContrast() → fallback chain (→MEDIUM→HIGH)
    5. Trả về (rect, box, qr_text, qr_points_180, qr_points, strategy_used)
    
    Args:
        src: BGR image (np.ndarray)
        threshold_value: Threshold cho binary (không dùng nữa vì Otsu adaptive)
    
    Returns:
        tuple: (rect, box, qr_text, qr_points_180, qr_points, strategy_used)
               hoặc (None, None, None, None, None, None) nếu thất bại
    """
    
    if src is None or src.size == 0:
        return (None, None, None, None, None, None)
    
    try:
        # 1. PREPROCESSING
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. TẦNG 1: Phân tích
        analysis = analyze_frame(gray)
        
        # 3. Log analysis
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║           FRAME ANALYSIS - AUTO CONTRAST DETECTION            ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print(f"║  📊 Final Score:     {analysis.final_score:6.3f}                              ║")
        print(f"║  🎯 Strategy Level:  {analysis.level:<10}                        ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("║  METRICS BREAKDOWN:                                            ║")
        print(f"║    • Separation:     {analysis.separation:6.3f}  (peaks: {analysis.peak1_position:3}, {analysis.peak2_position:3})       ║")
        print(f"║    • Edge Strength:  {analysis.edge_strength:6.3f}  ({analysis.edge_pixel_count:5} pixels)         ║")
        print(f"║    • Contrast Ratio: {analysis.contrast_ratio:6.3f}  (σ={analysis.stddev_intensity:6.1f})           ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        
        # 4. TẦNG 2: Routing và Fallback
        result = None
        strategy_used = ""
        
        if analysis.level == 'High':
            print("🟢 Executing Strategy: HIGH CONTRAST")
            result = detect_with_high_contrast(src, gray, threshold_value)
            print(f"   Result: {'✅ SUCCESS' if result[0] is not None else '❌ FAILED'}")
            if result[0] is not None:
                strategy_used = "HIGH"
        
        elif analysis.level == 'Medium':
            print("🟡 Executing Strategy: MEDIUM CONTRAST")
            result = detect_with_medium_contrast(src, gray)
            print(f"   Result: {'✅ SUCCESS' if result[0] is not None else '❌ FAILED'}")
            
            if result[0] is not None:
                strategy_used = "MEDIUM"
            else:
                # Fallback to HIGH
                print("⚠️  MEDIUM failed, falling back to HIGH strategy...")
                result = detect_with_high_contrast(src, gray, threshold_value)
                print(f"   Fallback Result: {'✅ SUCCESS' if result[0] is not None else '❌ FAILED'}")
                if result[0] is not None:
                    strategy_used = "MEDIUM→HIGH"
        
        elif analysis.level == 'Low':
            print("🔴 Executing Strategy: LOW CONTRAST (QR-First)")
            
            # Apply CLAHE preprocessing (như trong C#)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            print("  → Applied CLAHE preprocessing (adaptive contrast enhancement)")
            
            result = detect_with_low_contrast(src, gray)
            print(f"   Result: {'✅ SUCCESS' if result[0] is not None else '❌ FAILED'}")
            
            if result[0] is not None:
                strategy_used = "LOW"
            else:
                # Fallback to MEDIUM
                print("⚠️  LOW failed, falling back to MEDIUM strategy...")
                result = detect_with_medium_contrast(src, gray)
                print(f"   Fallback Result: {'✅ SUCCESS' if result[0] is not None else '❌ FAILED'}")
                
                if result[0] is not None:
                    strategy_used = "LOW→MEDIUM"
                else:
                    # Fallback to HIGH
                    print("⚠️  MEDIUM failed, falling back to HIGH strategy...")
                    result = detect_with_high_contrast(src, gray, threshold_value)
                    print(f"   Final Fallback Result: {'✅ SUCCESS' if result[0] is not None else '❌ FAILED'}")
                    if result[0] is not None:
                        strategy_used = "LOW→MEDIUM→HIGH"
        
        else:
            print("❌ ERROR: Unknown contrast level")
            result = (None, None, None, None, None)
            strategy_used = "ERROR"
        
        # 5. Log final result
        if result[0] is not None:
            qr_text = result[2] if result[2] else "N/A"
            print(f"✅ FINAL RESULT: Label detected | QR: {qr_text} | Strategy: {strategy_used}")
        else:
            print("❌ FINAL RESULT: Label NOT detected")
        print("")
        
        # 6. Return with strategy_used
        return (*result, strategy_used)
    
    except Exception as e:
        print(f"[DetectLabelRegion ERROR] {e}")
        import traceback
        traceback.print_exc()
        return (None, None, None, None, None, None)
```

---

## 📂 Cấu trúc thư mục Python project

```
label-detector/
├── src/
│   ├── __init__.py
│   ├── label_region_extractor.py    # Core logic (TẦNG 1 + TẦNG 2)
│   └── utils.py                      # Helper functions (visualization, etc.)
│
├── tests/
│   ├── __init__.py
│   └── test_label_detector.py       # Unit tests
│
├── data/
│   ├── test_images/                 # Ảnh test
│   │   ├── black_shirt.jpg
│   │   ├── white_shirt.jpg
│   │   ├── gray_shirt.jpg
│   │   └── ...
│   └── results/                     # Output visualization
│
├── main.py                          # Script chạy test trên ảnh
├── requirements.txt                 # Dependencies
└── PYTHON_IMPLEMENTATION_PLAN.md    # File này
```

---

## 🔧 Dependencies (requirements.txt)

```
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0  # For QRCodeDetector
numpy>=1.24.0
```

---

## 🧪 Test Script (main.py)

```python
import cv2
import numpy as np
from pathlib import Path
from src.label_region_extractor import detect_label_region

def visualize_result(src, result):
    """Vẽ kết quả lên ảnh."""
    rect, box, qr_text, qr_points_180, qr_points, strategy_used = result
    
    if rect is None:
        # Draw "NOT DETECTED" text
        vis = src.copy()
        cv2.putText(vis, "NOT DETECTED", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return vis
    
    vis = src.copy()
    
    # Draw label box
    cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
    
    # Draw QR points
    if qr_points is not None:
        for i, pt in enumerate(qr_points):
            pt = tuple(pt.astype(int))
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
            cv2.putText(vis, str(i), pt, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw info text
    info = f"Strategy: {strategy_used} | QR: {qr_text if qr_text else 'N/A'}"
    cv2.putText(vis, info, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return vis


def main():
    """Test detection trên ảnh mẫu."""
    
    # Load test image
    image_path = "data/test_images/test.jpg"
    src = cv2.imread(image_path)
    
    if src is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    print(f"✅ Loaded image: {src.shape}")
    print("")
    
    # Run detection
    result = detect_label_region(src)
    
    # Visualize
    vis = visualize_result(src, result)
    
    # Save result
    output_path = "data/results/output.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis)
    print(f"✅ Saved result to: {output_path}")
    
    # Display
    cv2.imshow("Detection Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 📊 Chi tiết khác biệt so với kế hoạch ban đầu

### 1. **Thresholds đã điều chỉnh**
| Parameter | Kế hoạch cũ | Mã C# thực tế | Lý do |
|-----------|-------------|---------------|--------|
| HIGH_THRESHOLD | 0.6 | 0.45 | Nới rộng để áo đen vào HIGH |
| MEDIUM_THRESHOLD | 0.3 | 0.25 | Thu hẹp để tránh overlap |
| LABEL_WIDTH_RATIO | 9.0 | 4.0 | Frame không quá dài |

### 2. **Strategy HIGH: Otsu Threshold**
- Kế hoạch: Hardcoded threshold (150)
- Thực tế: **Otsu adaptive threshold** (tự động)
- Lợi ích: Thích ứng với mọi màu áo (đen, trắng, xám...)

### 3. **Strategy MEDIUM: Canny thresholds**
- Kế hoạch: 50/150
- Thực tế: **30/100** (lower thresholds)
- Lý do: Nhạy hơn với edge trên nền tối/sáng

### 4. **Strategy LOW: CLAHE preprocessing**
- Kế hoạch: Chỉ có Histogram Equalization
- Thực tế: **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Lợi ích: Cải thiện contrast cục bộ, hoạt động tốt với mọi màu áo

### 5. **Fallback Chain đầy đủ**
- HIGH: Không có fallback
- MEDIUM: → HIGH
- LOW: → MEDIUM → HIGH (2-level fallback)

---

## 🚀 Lộ trình triển khai

### Phase 1: Setup project (30 phút)
- [x] Tạo cấu trúc thư mục
- [ ] Tạo `requirements.txt`
- [ ] Tạo file `__init__.py`
- [ ] Setup environment (`python -m venv venv`)

### Phase 2: Implement TẦNG 1 (2 giờ)
- [ ] Tạo `ContrastAnalysisResult` dataclass
- [ ] Implement `analyze_histogram()`
- [ ] Implement `analyze_edges()`
- [ ] Implement `analyze_contrast()`
- [ ] Implement `analyze_frame()`
- [ ] Test với 1 ảnh mẫu, kiểm tra metrics

### Phase 3: Implement Strategy HIGH (1 giờ)
- [ ] Implement `detect_with_high_contrast()`
- [ ] Test với ảnh áo đen

### Phase 4: Implement Strategy MEDIUM (1.5 giờ)
- [ ] Implement `detect_with_medium_contrast()`
- [ ] Test với ảnh áo màu nhạt

### Phase 5: Implement Strategy LOW (2 giờ)
- [ ] Implement `detect_with_low_contrast()`
- [ ] Test với ảnh áo trắng/kem

### Phase 6: Tích hợp (1 giờ)
- [ ] Implement `detect_label_region()` với routing + fallback
- [ ] Test end-to-end với 10 ảnh

### Phase 7: Visualization & Debug (1 giờ)
- [ ] Implement `visualize_result()`
- [ ] Tạo `main.py`
- [ ] Test batch processing

**Tổng thời gian:** ~9 giờ

---

## 🐛 Debugging Tips

### 1. Kiểm tra metrics không ổn định
- In ra histogram smoothed array
- Visualize Canny edges
- Plot metrics qua nhiều frames

### 2. Strategy detection sai
- Log từng bước trong strategies
- Visualize binary/morph/edges
- Kiểm tra contour filtering

### 3. QR detection fail
- Kiểm tra ROI crop có đúng không
- Test QR detector riêng
- Thử preprocessing khác (CLAHE, equalizeHist)

### 4. Geometry inference sai (Strategy LOW)
- In ra QR points
- Visualize vectors (dir_right, dir_down)
- Kiểm tra LABEL_WIDTH_RATIO

---

## 📝 Notes quan trọng

### 1. **RotatedRect trong Python**
OpenCV Python không có struct RotatedRect như C#. Thay vào đó:
```python
# C#: RotatedRect rect = Cv2.MinAreaRect(contour);
# Python: rect = cv2.minAreaRect(contour)
#         → returns tuple: ((cx, cy), (w, h), angle)

rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)  # Convert to 4 corners
box = np.int0(box)
```

### 2. **QRCodeDetector output**
```python
# C#: qr.DetectAndDecode(image, out points, straight)
# Python: text, points, straight = qr.detectAndDecode(image)

# points shape: (4, 2) or (1, 4, 2) - cần reshape!
if points is not None and points.ndim == 3:
    points = points.reshape(-1, 2)
```

### 3. **Logging format**
Giữ nguyên format box-drawing characters (╔═╗║╚╝) để dễ đối chiếu với C#.

### 4. **Coordinate systems**
- C# OpenCvSharp: `Point(x, y)`
- Python OpenCV: `(x, y)` tuple hoặc `np.array([x, y])`
- Lưu ý: NumPy array indexing: `img[y, x]` (row, col)

---

## ✅ Checklist hoàn thành

### Core Implementation
- [ ] TẦNG 1: 4 hàm phân tích
- [ ] Strategy HIGH
- [ ] Strategy MEDIUM
- [ ] Strategy LOW
- [ ] Hàm chính `detect_label_region()`
- [ ] Fallback chain logic

### Testing
- [ ] Test từng strategy riêng lẻ
- [ ] Test fallback chain
- [ ] Test với 30 ảnh (10 mỗi loại áo)
- [ ] Đo accuracy

### Visualization
- [ ] Vẽ label box
- [ ] Vẽ QR points
- [ ] Display metrics
- [ ] Save results

### Documentation
- [ ] Docstrings cho mọi hàm
- [ ] Comments giải thích logic phức tạp
- [ ] README.md hướng dẫn sử dụng

---

**Phiên bản:** 1.0 (From C# Source Code)  
**Ngày:** November 13, 2025  
**Trạng thái:** Ready for Implementation  
**Nguồn:** `LabelRegionExtractor.cs` (production code)

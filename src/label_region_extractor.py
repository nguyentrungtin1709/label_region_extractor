"""
Label Region Extractor - Python Implementation
Chuyển đổi từ C# LabelRegionExtractor.cs

Hệ thống 3 tầng:
- TẦNG 1: Analysis (phân tích độ tương phản)
- TẦNG 2: Strategies (3 chiến lược: HIGH/MEDIUM/LOW)
- Fallback Chain: MEDIUM→HIGH, LOW→MEDIUM→HIGH
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================================
# CONSTANTS (từ C# LabelRegionExtractor.cs)
# ============================================================================

# Analysis thresholds
HIGH_THRESHOLD = 0.45      # Nới rộng từ 0.6 → 0.45
MEDIUM_THRESHOLD = 0.25    # Thu hẹp từ 0.3 → 0.25
EDGE_MAX = 0.1             # Normalization cho edge strength

# Label expansion ratios (cho Strategy LOW)
LABEL_WIDTH_RATIO = 4.0    # Giảm từ 9.0 → 4.0
LABEL_HEIGHT_RATIO = 3.0   # Giữ nguyên 3×

# Debug output directory
DEBUG_OUTPUT_DIR = "data/debug"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_analysis_region(gray: np.ndarray) -> np.ndarray:
    """
    Lấy vùng để phân tích.
    
    Trước: Lấy 1/3 center (vì ảnh lớn, nhãn ở giữa)
    Hiện tại: Lấy toàn bộ ảnh (vì input đã là vùng nhỏ chứa nhãn)
    
    Args:
        gray: Ảnh grayscale
    
    Returns:
        Region để phân tích (toàn bộ ảnh)
    """
    return gray


def save_debug_image(image: np.ndarray, filename: str, cmap='gray'):
    """
    Lưu ảnh debug.
    
    Args:
        image: Ảnh cần lưu
        filename: Tên file (sẽ tự động thêm prefix debug_)
        cmap: Colormap cho matplotlib
    """
    output_dir = Path(DEBUG_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"debug_{filename}"
    
    if len(image.shape) == 2:  # Grayscale
        plt.figure(figsize=(10, 6))
        plt.imshow(image, cmap=cmap)
        plt.colorbar()
        plt.title(filename)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:  # Color (BGR -> RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  💾 Saved debug: {output_path}")


def save_debug_json(data: dict, filename: str):
    """
    Lưu dữ liệu debug dạng JSON.
    
    Args:
        data: Dictionary chứa dữ liệu
        filename: Tên file (sẽ tự động thêm prefix debug_)
    """
    output_dir = Path(DEBUG_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"debug_{filename}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved debug: {output_path}")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

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


# ============================================================================
# TẦNG 1: ANALYSIS METHODS
# ============================================================================

def analyze_histogram(gray: np.ndarray) -> Tuple[int, int, float]:
    """
    Phân tích histogram để tìm 2 peaks chính và tính separation.
    
    Logic:
    1. Lấy vùng phân tích (toàn bộ ảnh)
    2. Tính histogram 256 bins
    3. Smooth bằng moving average 5 bins
    4. Tìm local maxima (> 0.5×avgHeight, cách nhau >30 bins)
    5. Chọn 2 peaks cao nhất, sort theo position
    6. separation = |peak2 - peak1| / 255.0
    
    Returns:
        (peak1_pos, peak2_pos, separation)
    """
    # 1. Lấy vùng phân tích
    analysis_roi = get_analysis_region(gray)
    
    # 2. Histogram
    hist = cv2.calcHist([analysis_roi], [0], None, [256], [0, 256])
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
        peak1, peak2, separation = 0, 255, 0.0
    else:
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
        peaks = sorted(peaks, key=lambda x: x[0])  # Sort by position
        
        peak1 = peaks[0][0]
        peak2 = peaks[1][0]
        separation = abs(peak2 - peak1) / 255.0
    
    # Debug: Vẽ histogram với peaks (luôn vẽ dù có peaks hay không)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist, color='gray', alpha=0.5, label='Original')
    plt.plot(smoothed, color='blue', label='Smoothed')
    plt.axvline(peak1, color='red', linestyle='--', label=f'Peak1={peak1}')
    plt.axvline(peak2, color='green', linestyle='--', label=f'Peak2={peak2}')
    plt.title(f'Histogram Analysis (Separation={separation:.3f})')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.imshow(analysis_roi, cmap='gray')
    plt.title('Analysis Region')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = Path(DEBUG_OUTPUT_DIR) / "debug_01_histogram.png"
    Path(DEBUG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved debug: {output_path}")
    
    return (peak1, peak2, separation)


def analyze_edges(gray: np.ndarray) -> Tuple[int, float]:
    """
    Phân tích edges bằng Canny để tính edge strength.
    
    Logic:
    1. Lấy vùng phân tích (toàn bộ ảnh)
    2. Canny(50, 150)
    3. Đếm non-zero pixels
    4. edge_strength = edge_pixels / total_pixels
    
    Returns:
        (edge_pixels, edge_strength)
    """
    # 1. Lấy vùng phân tích
    analysis_roi = get_analysis_region(gray)
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(analysis_roi, threshold1=50, threshold2=150)
    
    # 3. Count edge pixels
    edge_pixels = cv2.countNonZero(edges)
    total_pixels = analysis_roi.shape[0] * analysis_roi.shape[1]
    
    edge_strength = edge_pixels / total_pixels
    
    # Debug: Lưu ảnh edges
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(analysis_roi, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Canny Edges (Strength={edge_strength:.4f}, {edge_pixels} pixels)')
    plt.axis('off')
    
    plt.tight_layout()
    output_path = Path(DEBUG_OUTPUT_DIR) / "debug_02_edges.png"
    Path(DEBUG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved debug: {output_path}")
    
    return (edge_pixels, edge_strength)


def analyze_contrast(gray: np.ndarray) -> Tuple[float, float, float]:
    """
    Phân tích contrast bằng standard deviation.
    
    Logic:
    1. Lấy vùng phân tích (toàn bộ ảnh)
    2. Tính mean, stddev
    3. contrast_ratio = stddev / 128.0
    
    Returns:
        (mean, stddev, contrast_ratio)
    """
    # 1. Lấy vùng phân tích
    analysis_roi = get_analysis_region(gray)
    
    # 2. Calculate mean and stddev
    mean, stddev = cv2.meanStdDev(analysis_roi)
    mean = mean[0][0]
    stddev = stddev[0][0]
    
    # 3. Contrast ratio
    contrast_ratio = stddev / 128.0
    
    # Debug: Lưu JSON và visualization
    contrast_data = {
        "mean_intensity": float(mean),
        "stddev_intensity": float(stddev),
        "contrast_ratio": float(contrast_ratio),
        "min_intensity": float(np.min(analysis_roi)),
        "max_intensity": float(np.max(analysis_roi)),
        "image_shape": list(analysis_roi.shape)
    }
    
    save_debug_json(contrast_data, "03_contrast.json")
    
    # Vẽ phân bố cường độ
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(analysis_roi, cmap='gray')
    plt.title(f'Analysis Region (Mean={mean:.1f}, StdDev={stddev:.1f})')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.hist(analysis_roi.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean={mean:.1f}')
    plt.axvline(mean - stddev, color='orange', linestyle=':', label=f'Mean-σ={mean-stddev:.1f}')
    plt.axvline(mean + stddev, color='orange', linestyle=':', label=f'Mean+σ={mean+stddev:.1f}')
    plt.title(f'Intensity Distribution (Contrast Ratio={contrast_ratio:.3f})')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(DEBUG_OUTPUT_DIR) / "debug_03_contrast.png"
    Path(DEBUG_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved debug: {output_path}")
    
    return (mean, stddev, contrast_ratio)


def analyze_frame(gray: np.ndarray) -> ContrastAnalysisResult:
    """
    Phân tích frame để tính final score và xác định contrast level.
    
    Logic từ C#:
    1. Gọi 3 hàm phân tích
    2. Normalize edge_strength (min(edge_strength / 0.1, 1.0))
    3. Final Score = separationx0.4 + edge_strength_normx0.3 + contrast_ratiox0.3
    4. Phân loại: >0.45=High, 0.25-0.45=Medium, <0.25=Low
    
    Returns:
        ContrastAnalysisResult
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


# ============================================================================
# TẦNG 2: STRATEGY METHODS
# ============================================================================

def detect_with_high_contrast(src: np.ndarray, gray: np.ndarray, 
                             threshold_value: int = 150) -> Tuple:
    """
    Strategy HIGH: Binary Threshold + Morphology (cho áo tối/màu đậm).
    
    Logic từ C# (đã cập nhật):
    1. Otsu adaptive threshold thay vì hardcoded value
    2. Morphology: Open(3x3, 1 iter) → Close(3x3, 2 iters)
    3. FindContours(EXTERNAL)
    4. Chọn largest contour theo area
    5. MinAreaRect → box
    6. Crop bounding rect → QR verification
    7. Trả về (rect, box, qr_text, qr_points_180, qr_points)
    
    Returns:
        tuple: (rect, box, qr_text, qr_points_180, qr_points) or (None, None, None, None, None)
    """
    print("  → Method: Binary Threshold + Morphology")
    
    # 1. Otsu adaptive threshold
    otsu_threshold, binary = cv2.threshold(gray, 0, 255, 
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f"  → Otsu adaptive threshold: {otsu_threshold:.1f} (auto-calculated)")
    
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
    box = np.int32(box)
    
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
    if qr_points_180 is not None and len(qr_points_180) > 0:
        # Reshape if needed (sometimes shape is (1, 4, 2))
        if qr_points_180.ndim == 3:
            qr_points_180 = qr_points_180.reshape(-1, 2)
        
        # Convert to global coordinates
        qr_points = qr_points_180.copy()
        qr_points[:, 0] += x
        qr_points[:, 1] += y
    
    if qr_text:
        print(f"  ✓ QR detected: {qr_text}")
        return (rect, box, qr_text, qr_points_180, qr_points)
    
    print("  ✗ No QR code found in label region")
    return (None, None, None, None, None)


def detect_with_medium_contrast(src: np.ndarray, gray: np.ndarray) -> Tuple:
    """
    Strategy MEDIUM: Canny Edge + Strong Morphology (cho áo màu nhạt).
    
    Logic từ C# (đã cập nhật):
    1. Canny(30, 100) - Lower thresholds để nhạy hơn
    2. Morphology: Close(7x7, 3 iters) → Dilate(7x7, 1 iter)
    3. FindContours(EXTERNAL)
    4. Filter CHỈ theo area ratio (5-80%)
    5. Sort theo area (lớn nhất trước)
    6. Loop candidates → verify QR → Early exit khi tìm thấy
    
    Returns:
        tuple: (rect, box, qr_text, qr_points_180, qr_points) or (None, None, None, None, None)
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
        if qr_points_180 is not None and len(qr_points_180) > 0:
            if qr_points_180.ndim == 3:
                qr_points_180 = qr_points_180.reshape(-1, 2)
            
            qr_points = qr_points_180.copy()
            qr_points[:, 0] += x
            qr_points[:, 1] += y
        
        if qr_text:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            print(f"  ✓ QR detected in candidate (area={area:.0f}): {qr_text}")
            return (rect, box, qr_text, qr_points_180, qr_points)
    
    print("  ✗ No QR found in any candidate")
    return (None, None, None, None, None)


def detect_with_low_contrast(src: np.ndarray, gray: np.ndarray) -> Tuple:
    """
    Strategy LOW: QR-First + Geometry Inference (cho áo trắng/kem).
    
    Logic từ C# (đã cập nhật):
    1. Histogram equalization cho QR detection robustness
    2. Detect QR trên enhanced image (fallback to original nếu fail)
    3. Tính geometry QR (vectors, width, height, angle)
    4. Suy luận label với expansion ratios (4.0x, 3.0x)
    5. Construct 4 corners (QR ở TRÁI DƯỚI, expand PHẢI + TRÊN)
    6. Tạo RotatedRect từ 4 corners
    
    Returns:
        tuple: (rect, box, qr_text, None, qr_points)
        Note: qr_points_180 = None vì detect trên toàn ảnh
    """
    print("  → Method: QR-First + Geometry Inference")
    
    # 1. Enhance contrast cho QR detection
    enhanced = cv2.equalizeHist(gray)
    print("  → Applied histogram equalization for QR detection robustness")
    
    # 2. Detect QR
    qr_detector = cv2.QRCodeDetector()
    
    # Try on enhanced first
    qr_text, qr_points, _ = qr_detector.detectAndDecode(enhanced)
    
    # Fallback to original
    if not qr_text:
        qr_text, qr_points, _ = qr_detector.detectAndDecode(src)
    
    if not qr_text or qr_points is None or len(qr_points) < 4:
        print("  ✗ No QR code detected")
        return (None, None, None, None, None)
    
    # Reshape if needed
    if qr_points.ndim == 3:
        qr_points = qr_points.reshape(-1, 2)
    
    print(f"  ✓ QR detected: {qr_text}")
    
    # 3. Calculate QR geometry
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
    
    angle_rad = np.arctan2(top_vec[1], top_vec[0])
    angle_deg = angle_rad * 180.0 / np.pi
    print(f"  → QR geometry: {qr_width:.1f}x{qr_height:.1f} px, angle={angle_deg:.1f}°")
    
    # 4. Infer label dimensions
    label_width = qr_width * LABEL_WIDTH_RATIO
    label_height = qr_height * LABEL_HEIGHT_RATIO
    print(f"  → Predicted label: {label_width:.1f}x{label_height:.1f} px")
    print(f"  → Expansion: width={LABEL_WIDTH_RATIO}xQR, height={LABEL_HEIGHT_RATIO}xQR")
    
    # 5. Calculate 4 corners
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
    
    # 6. Create RotatedRect
    label_center = (label_top_left + label_top_right + 
                   label_bottom_right + label_bottom_left) / 4.0
    
    angle = angle_deg
    
    # RotatedRect as tuple (OpenCV format)
    rect = (tuple(label_center), (label_width, label_height), angle)
    
    box = np.array([label_top_left, label_top_right, 
                    label_bottom_right, label_bottom_left], dtype=np.int32)
    
    print(f"  ✓ Label constructed: center=({label_center[0]:.1f},{label_center[1]:.1f}), angle={angle:.1f}°")
    
    # qr_points_180 = None (detect trên toàn ảnh, không có ROI cục bộ)
    return (rect, box, qr_text, None, qr_points)


# ============================================================================
# HÀM CHÍNH: DETECT LABEL REGION
# ============================================================================

def detect_label_region(src: np.ndarray, 
                       threshold_value: int = 150) -> Tuple[Optional[Tuple], 
                                                            Optional[np.ndarray], 
                                                            Optional[str],
                                                            Optional[np.ndarray],
                                                            Optional[np.ndarray],
                                                            Optional[str]]:
    """
    Phát hiện vùng nhãn trong ảnh.
    
    Logic từ C# (flow chính xác):
    1. Preprocessing: BGR → Gray → GaussianBlur(5x5)
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

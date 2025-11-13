"""
Test Script - Label Region Extractor
Chạy detection trên ảnh và hiển thị kết quả

Usage:
    python main.py <input_image> <output_image>
    
Example:
    python main.py data/test_images/test.jpg data/results/output.jpg
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from label_region_extractor import detect_label_region


def visualize_result(src: np.ndarray, result: tuple) -> np.ndarray:
    """
    Vẽ kết quả detection lên ảnh.
    
    Args:
        src: Ảnh gốc (BGR)
        result: Tuple từ detect_label_region()
                (rect, box, qr_text, qr_points_180, qr_points, strategy_used)
    
    Returns:
        Ảnh đã vẽ visualization
    """
    rect, box, qr_text, qr_points_180, qr_points, strategy_used = result
    
    if rect is None:
        # Draw "NOT DETECTED" text
        vis = src.copy()
        cv2.putText(vis, "NOT DETECTED", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(vis, f"Strategy: {strategy_used if strategy_used else 'N/A'}", 
                   (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis
    
    vis = src.copy()
    
    # Draw label box (green)
    cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
    
    # Draw corner points
    for i, pt in enumerate(box):
        cv2.circle(vis, tuple(pt), 5, (255, 0, 0), -1)
        cv2.putText(vis, str(i), tuple(pt + [10, 10]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw QR points (red)
    if qr_points is not None and len(qr_points) > 0:
        for i, pt in enumerate(qr_points):
            pt_int = tuple(pt.astype(int))
            cv2.circle(vis, pt_int, 5, (0, 0, 255), -1)
            cv2.putText(vis, f"QR{i}", pt_int, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw QR box
        qr_box = qr_points.astype(np.int32)
        cv2.polylines(vis, [qr_box], True, (0, 0, 255), 2)
    
    # Draw info text
    # Thay thế các ký tự Unicode không hỗ trợ bằng ASCII
    strategy_display = strategy_used.replace('→', '->')
    
    info_lines = [
        f"Strategy: {strategy_display}",
        f"QR: {qr_text if qr_text else 'N/A'}"
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(vis, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    # Draw rect center and angle
    if rect is not None:
        center = (int(rect[0][0]), int(rect[0][1]))
        angle = rect[2]
        cv2.circle(vis, center, 3, (255, 0, 255), -1)
        cv2.putText(vis, f"Angle: {angle:.1f}", 
                   (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return vis


def main(input_image=None, output_dir=None):
    """Test detection trên ảnh mẫu."""
    
    # Nếu không truyền tham số, parse từ command line
    if input_image is None:
        parser = argparse.ArgumentParser(
            description='Label Region Extractor - Detect and extract label regions from images',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
    python main.py data/test_images/test.jpg data/results
    python main.py input.png output_folder
        """
        )
        parser.add_argument('input_image', type=str, help='Path to input image')
        parser.add_argument('output_dir', type=str, nargs='?', default='data/results', 
                          help='Output directory (default: data/results)')
        
        args = parser.parse_args()
        
        image_path = args.input_image
        output_directory = args.output_dir
    else:
        # Sử dụng tham số được truyền vào
        image_path = input_image
        output_directory = output_dir if output_dir else 'data/results'
    
    # Tạo tên file output từ tên file input
    input_filename = Path(image_path).stem  # Lấy tên file không có extension
    output_path = Path(output_directory) / f"output_{input_filename}.png"
    
    print("=" * 70)
    print("LABEL REGION EXTRACTOR - TEST SCRIPT")
    print("=" * 70)
    print()
    
    # Load test image
    print(f"📁 Loading image: {image_path}")
    src = cv2.imread(image_path)
    
    if src is None:
        print(f"❌ Cannot load image: {image_path}")
        print(f"   File not found or invalid format")
        return
    
    print(f"✅ Loaded image: {src.shape[1]}×{src.shape[0]} px")
    print()
    
    # Run detection
    print("🔍 Running detection...")
    print()
    result = detect_label_region(src)
    
    # Visualize
    print("🎨 Creating visualization...")
    vis = visualize_result(src, result)
    
    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"✅ Saved result to: {output_path.absolute()}")
    print()
    
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main("data/test_images/011-white-guide-box.png", "data/results")

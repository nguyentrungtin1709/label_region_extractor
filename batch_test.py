"""
Batch Test Script - Test detection trên nhiều ảnh
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from label_region_extractor import detect_label_region


def batch_test():
    """Test detection trên tất cả ảnh trong test_images."""
    
    test_dir = Path("data/test_images")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả ảnh
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No test images found in data/test_images/")
        print("   Please add some test images (.jpg or .png)")
        return
    
    print("=" * 70)
    print(f"BATCH TEST - {len(image_files)} images")
    print("=" * 70)
    print()
    
    results = []
    
    for img_path in image_files:
        print(f"📁 Testing: {img_path.name}")
        
        # Load image
        src = cv2.imread(str(img_path))
        if src is None:
            print(f"   ❌ Cannot load image")
            continue
        
        # Run detection
        result = detect_label_region(src)
        rect, box, qr_text, qr_points_180, qr_points, strategy_used = result
        
        # Store result
        success = rect is not None
        results.append({
            'filename': img_path.name,
            'success': success,
            'strategy': strategy_used,
            'qr_text': qr_text if qr_text else 'N/A'
        })
        
        # Visualize and save
        vis = src.copy()
        
        if success:
            # Draw label box
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
            
            # Draw QR points
            if qr_points is not None:
                for pt in qr_points:
                    pt_int = tuple(pt.astype(int))
                    cv2.circle(vis, pt_int, 5, (0, 0, 255), -1)
            
            # Info text
            cv2.putText(vis, f"Strategy: {strategy_used}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"QR: {qr_text if qr_text else 'N/A'}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "NOT DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(vis, f"Strategy: {strategy_used}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save
        output_path = results_dir / f"result_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), vis)
        
        print(f"   {'✅' if success else '❌'} {strategy_used} | QR: {qr_text if qr_text else 'N/A'}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"✅ Success: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print()
    
    # Strategy breakdown
    from collections import Counter
    strategies = [r['strategy'] for r in results if r['success']]
    strategy_counts = Counter(strategies)
    
    print("Strategy usage:")
    for strategy, count in strategy_counts.most_common():
        print(f"  • {strategy}: {count} times")
    print()
    
    # Failed images
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"❌ Failed images ({len(failed)}):")
        for r in failed:
            print(f"  • {r['filename']} (strategy: {r['strategy']})")
    
    print()
    print(f"📁 Results saved to: {results_dir.absolute()}")
    print()


if __name__ == "__main__":
    batch_test()

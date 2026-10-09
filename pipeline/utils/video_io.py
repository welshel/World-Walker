"""Small video probes with lazy optional backend imports."""
from __future__ import annotations

def decoded_frame_count(path):
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(str(path))
        try:
            return sum(1 for _ in reader)
        finally:
            reader.close()
    except ImportError:
        import cv2
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        capture.release()
        return count

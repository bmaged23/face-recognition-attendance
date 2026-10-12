"""English labels, using OpenCV's built-in font."""


def annotate(frame, matches, *, show_confidence=True):
    import cv2
    height, width = frame.shape[:2]
    for match in matches:
        x1, y1, x2, y2 = match.bbox
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        color = (0, 190, 0) if match.person else (0, 170, 255)
        label = match.person.name if match.person else "Unknown"
        if match.person and show_confidence:
            label += f' ({match.confidence * 100:.1f}%)'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return frame

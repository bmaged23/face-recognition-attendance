"""Rolling list of confirmed attendance names, drawn on the left of a video."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PanelLayout:
    margin: int
    width: int
    header_height: int
    first_row: int
    row_height: int
    gap: int
    capacity: int
    font_scale: float


def panel_layout(width, height):
    margin = max(2, min(12, width // 40, height // 40))
    panel_width = max(1, min(220, int(width * 0.3), width - 2 * margin))
    header_height = max(8, min(40, height // 6))
    row_height = max(8, min(34, height // 12))
    gap = max(2, min(6, height // 100))
    first_row = margin + header_height + gap
    capacity = max(0, (height - margin - first_row + gap) // (row_height + gap))
    return PanelLayout(margin, panel_width, header_height, first_row,
                       row_height, gap, capacity, min(0.65, row_height / 52))


def visible_names(records, capacity):
    # Records arrive once per person ID in first-confirmation order.
    return [row['name'] for row in records[-capacity:]] if capacity else []


def annotate_attendance(frame, records, action):
    if action not in {'checkin', 'checkout'}:
        raise ValueError('Attendance panel requires checkin or checkout')
    import cv2
    height, width = frame.shape[:2]
    layout = panel_layout(width, height)

    def draw_row(text, top, row_height, header=False):
        left = layout.margin
        right = min(width - 1, left + layout.width)
        bottom = min(height - 1, top + row_height - 1)
        if bottom < top:
            return
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), -1)
        padding = max(2, min(10, layout.width // 12))
        scale = layout.font_scale
        available = max(1, layout.width - 2 * padding)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1 if row_height < 25 else 2
        # Fit long names inside their black rectangle.
        label = text
        while label and cv2.getTextSize(label, font, scale, thickness)[0][0] > available:
            text = text[:-1]
            label = text + '...' if text else ''
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        x = left + padding
        y = min(bottom - max(1, baseline), top + (row_height + text_height) // 2)
        cv2.putText(frame, label, (x, y), font, scale,
                    (0, 220, 255) if header else (255, 255, 255), thickness, cv2.LINE_AA)

    draw_row('Check-in' if action == 'checkin' else 'Check-out',
             layout.margin, layout.header_height, header=True)
    for index, name in enumerate(visible_names(records, layout.capacity)):
        draw_row(name, layout.first_row + index * (layout.row_height + layout.gap), layout.row_height)
    return frame

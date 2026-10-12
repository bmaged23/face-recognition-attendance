import unittest
from unittest.mock import patch
import numpy as np
from face_recognition.video.attendance_panel import annotate_attendance, panel_layout, visible_names


class AttendancePanelTests(unittest.TestCase):
    def test_overflow_removes_oldest_and_keeps_newest_in_order(self):
        records = [{'id': str(i), 'name': f'Person {i}'} for i in range(6)]
        self.assertEqual(visible_names(records, 3), ['Person 3', 'Person 4', 'Person 5'])
        self.assertEqual(visible_names(records[:2], 3), ['Person 0', 'Person 1'])
        self.assertEqual(visible_names(records, 0), [])

    def test_header_and_each_name_have_black_rectangles(self):
        for action, heading in [('checkin', 'Check-in'), ('checkout', 'Check-out')]:
            frame = np.full((600, 800, 3), 180, dtype=np.uint8)
            with patch('cv2.putText') as text:
                annotate_attendance(frame, [{'name': 'Alex'}, {'name': 'Jordan'}], action)
            self.assertEqual([c.args[1] for c in text.call_args_list], [heading, 'Alex', 'Jordan'])
            layout = panel_layout(800, 600)
            self.assertEqual(frame[layout.margin + 1, layout.margin + 1].tolist(), [0, 0, 0])
            for index in range(2):
                y = layout.first_row + index * (layout.row_height + layout.gap)
                self.assertEqual(frame[y + 1, layout.margin + 1].tolist(), [0, 0, 0])
            self.assertEqual(frame[-1, -1].tolist(), [180, 180, 180])

    def test_full_column_renders_recent_names_below_fixed_header(self):
        layout = panel_layout(800, 600)
        records = [{'name': f'Name {i}'} for i in range(layout.capacity + 2)]
        with patch('cv2.putText') as text:
            annotate_attendance(np.zeros((600, 800, 3), dtype=np.uint8), records, 'checkin')
        self.assertEqual(text.call_args_list[0].args[1], 'Check-in')
        self.assertEqual(text.call_args_list[1].args[1], 'Name 2')
        self.assertEqual(text.call_args_list[-1].args[1], f'Name {layout.capacity + 1}')
        self.assertEqual(len(text.call_args_list), layout.capacity + 1)

    def test_long_names_fit_inside_rectangles(self):
        with patch('cv2.putText') as text:
            annotate_attendance(np.zeros((600, 800, 3), dtype=np.uint8),
                                [{'name': 'An Extremely Long Display Name That Must Fit'}], 'checkin')
        label = text.call_args_list[-1].args[1]
        self.assertTrue(label.endswith('...'))
        import cv2
        layout = panel_layout(800, 600)
        self.assertLessEqual(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, layout.font_scale, 2)[0][0], layout.width - 20)

    def test_layout_keeps_rows_inside_frame(self):
        for width, height in [(800, 600), (1920, 1080), (320, 240), (32, 32)]:
            layout = panel_layout(width, height)
            if layout.capacity:
                bottom = layout.first_row + (layout.capacity - 1) * (layout.row_height + layout.gap) + layout.row_height
                self.assertLessEqual(bottom, height - layout.margin)
            self.assertLessEqual(layout.margin + layout.width, width)

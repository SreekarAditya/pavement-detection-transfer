import unittest

import bharat


class BharatTests(unittest.TestCase):
    def test_frame_formats(self):
        pattern = "test/images/{}_frame_00006_jpg.rf." + "a" * 32 + ".jpg"
        for name in ("20250501091922_0060", "signal-2025-06-26-111944", "signal-2025-06-26-111944_002"):
            self.assertEqual(bharat.frame_key(pattern.format(name)), (name, 6))
        self.assertEqual(bharat.video_key("signal-2025-06-26-111944_002"), "signal-2025-06-26-111944")
        self.assertIsNone(bharat.frame_key(pattern.format("unknown")))

    def test_annotation_semantics(self):
        self.assertEqual(bharat.label_boxes(b"0 0 0.5 0.25 0.25\n0 0 0.5 0.25 0.25"), [(0, 240, 80, 400)])
        self.assertEqual(bharat.label_boxes(b""), [])
        with self.assertRaises(ValueError):
            bharat.label_boxes(b"1 0 0.5 0.25 0.25")

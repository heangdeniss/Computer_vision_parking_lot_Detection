"""Unit tests for server media and formatting helpers (no model weights needed)."""
import base64
import io
import unittest

import numpy as np
from PIL import Image

from server.detectors import counts_of
from server.media import decode_frame_bgr, draw_boxes, jpeg_url, load_image


class ServerHelpersTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 100), (255, 255, 255))
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        self.raw = buffer.getvalue()

    def test_load_image(self):
        image = load_image(self.raw)
        self.assertEqual((image.width, image.height), (200, 100))

    def test_decode_frame_bgr(self):
        bgr = decode_frame_bgr(self.raw)
        self.assertEqual(bgr.shape, (100, 200, 3))

    def test_jpeg_url(self):
        url = jpeg_url(self.image)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(base64.b64decode(url.split(",", 1)[1])), 0)

    def test_counts_of(self):
        detections = [{"class": "empty"}, {"class": "empty"}, {"class": "occupied"}]
        self.assertEqual(counts_of(detections, ["empty", "occupied"]), {"empty": 2, "occupied": 1})

    def test_draw_boxes_uses_green_and_red(self):
        detections = [
            {"class": "empty", "confidence": 0.9, "bbox": [10, 10, 70, 60]},
            {"class": "occupied", "confidence": 0.8, "bbox": [90, 10, 150, 60]},
        ]
        pixels = np.asarray(Image.open(io.BytesIO(
            base64.b64decode(draw_boxes(self.image, detections).split(",", 1)[1]))).convert("RGB"))
        green = np.sum((pixels[:, :, 1] > 200) & (pixels[:, :, 0] < 60) & (pixels[:, :, 2] < 60))
        red = np.sum((pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] < 60))
        self.assertGreater(green, 100)
        self.assertGreater(red, 100)


if __name__ == "__main__":
    unittest.main()

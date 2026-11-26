"""Unit tests for shared YOLO dataset parsing, metrics, and Faster R-CNN formatting."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from server.detectors import frcnn_detections_of
from src.data import YoloParkingDataset, parse_yolo_label_file
from src.metrics import match_predictions, update_confusion_matrix


class FasterRCNNHelpersTest(unittest.TestCase):
    def test_parse_yolo_label_file_converts_and_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "sample.txt"
            label.write_text(
                "0 0.5 0.5 0.4 0.2\n"
                "1 0.0 0.0 0.2 0.2\n"
                "0 0.5 0.5 0.0 0.2\n",
                encoding="utf-8",
            )
            boxes, classes = parse_yolo_label_file(label, width=1000, height=500)
            self.assertEqual(boxes.shape, (2, 4))
            self.assertEqual(classes.tolist(), [0, 1])
            np.testing.assert_allclose(boxes[0], [300.0, 200.0, 700.0, 300.0], atol=1e-4)
            np.testing.assert_allclose(boxes[1], [0.0, 0.0, 100.0, 50.0], atol=1e-4)

    def test_parse_missing_or_empty_label_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            boxes, classes = parse_yolo_label_file(Path(tmp) / "none.txt", 640, 480)
            self.assertEqual(boxes.shape, (0, 4))
            self.assertEqual(classes.shape, (0,))

    def test_dataset_shifts_labels_and_flips_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            Image.new("RGB", (200, 100), (120, 120, 120)).save(root / "images" / "lot.jpg")
            (root / "labels" / "lot.txt").write_text("0 0.2 0.3 0.2 0.4\n1 0.8 0.5 0.2 0.4\n")

            ds_no_flip = YoloParkingDataset(root / "images", ["empty", "occupied"], augment=False)
            tensor, target = ds_no_flip[0]
            self.assertEqual(tuple(tensor.shape), (3, 100, 200))
            self.assertEqual(target["labels"].tolist(), [1, 2])
            np.testing.assert_allclose(target["boxes"][0].numpy(), [20.0, 10.0, 60.0, 50.0], atol=1e-3)

            ds_flip = YoloParkingDataset(root / "images", ["empty", "occupied"], augment=True, fliplr=1.0)
            _, flipped = ds_flip[0]
            np.testing.assert_allclose(flipped["boxes"][0].numpy(), [140.0, 10.0, 180.0, 50.0], atol=1e-3)

    def test_frcnn_detections_of_remaps_and_filters(self):
        output = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0], [1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 9.0, 9.0]]),
            "scores": torch.tensor([0.92, 0.15, 0.88]),
            "labels": torch.tensor([2, 1, 0]),
        }
        dets = frcnn_detections_of(output, ["empty", "occupied"], conf=0.25)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["class"], "occupied")
        self.assertAlmostEqual(dets[0]["confidence"], 0.92, places=3)
        self.assertEqual(dets[0]["bbox"], [10.0, 20.0, 50.0, 60.0])

    def test_match_predictions_and_confusion_matrix(self):
        gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [200.0, 0.0, 300.0, 100.0]])
        gt_cls = torch.tensor([0, 1])
        pred_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [200.0, 0.0, 300.0, 100.0]])
        pred_cls = torch.tensor([0, 0])
        pred_scores = torch.tensor([0.9, 0.8])

        correct = match_predictions(pred_boxes, pred_cls, gt_boxes, gt_cls)
        self.assertTrue(bool(correct[0].all()))
        self.assertFalse(bool(correct[1].any()))

        cm = np.zeros((3, 3), dtype=np.int64)
        update_confusion_matrix(cm, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, conf_thres=0.25)
        self.assertEqual(int(cm[0, 0]), 1)
        self.assertEqual(int(cm[0, 1]), 1)


if __name__ == "__main__":
    unittest.main()

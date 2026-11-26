# Graph Report - Parking-lot Detection  (2026-11-26)

## Corpus Check
- 42 files · ~15,564 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 5, .css 1, .ipynb 1)

## Summary
- 388 nodes · 714 edges · 19 communities (13 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c0bce4bb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- detectors.py
- faster_rcnn/train.py
- App
- app.py
- package.json
- train_one_epoch
- faster_rcnn.yaml
- Parking Lot Occupancy Detection — YOLOv11 vs. Faster R-CNN
- inspect_model.py
- evaluate_detector
- Model & System Architecture — YOLOv11s vs. Faster R-CNN
- parking-detector
- parse_yolo_label_file
- resources
- install-docker-popos.sh

## God Nodes (most connected - your core abstractions)
1. `App()` - 24 edges
2. `YoloParkingDataset` - 17 edges
3. `evaluate_detector()` - 17 edges
4. `Parking Lot Occupancy Detection — YOLOv11 vs. Faster R-CNN` - 14 edges
5. `main()` - 14 edges
6. `load_faster_rcnn_checkpoint()` - 13 edges
7. `load_data_yaml()` - 12 edges
8. `draw_boxes()` - 12 edges
9. `eval_frcnn()` - 11 edges
10. `Model & System Architecture — YOLOv11s vs. Faster R-CNN` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Shared Annotation Adapter (`src/data.py`)` --references--> `parse_yolo_label_file()`  [INFERRED]
  docs/architecture.md → src/data.py
- `Shared Annotation Adapter (`src/data.py`)` --references--> `YoloParkingDataset`  [INFERRED]
  docs/architecture.md → src/data.py
- `FasterRCNNHelpersTest` --uses--> `YoloParkingDataset`  [INFERRED]
  tests/test_faster_rcnn.py → src/data.py
- `_ensure_faster_rcnn()` --calls--> `FasterRCNNDetector`  [EXTRACTED]
  server/app.py → server/detectors.py
- `eval_frcnn()` --indirect_call--> `detection_collate()`  [INFERRED]
  src/compare.py → src/data.py

## Import Cycles
- None detected.

## Communities (19 total, 6 thin omitted)

### Community 0 - "detectors.py"
Cohesion: 0.14
Nodes (5): counts_of(), detections_of(), FasterRCNNDetector, frcnn_detections_of(), YoloDetector

### Community 1 - "faster_rcnn/train.py"
Cohesion: 0.07
Nodes (18): eval_frcnn(), eval_yolo(), main(), save_comparison_chart(), detection_collate(), load_data_yaml(), YoloParkingDataset, main() (+10 more)

### Community 2 - "App"
Cohesion: 0.10
Nodes (27): App(), addFiles(), clear(), detectImages(), detectVideo(), onDrop(), onSubmit(), releaseUrls() (+19 more)

### Community 3 - "app.py"
Cohesion: 0.06
Nodes (17): compare(), _ensure_faster_rcnn(), frame(), health(), predict(), _predict_impl(), preview(), _rel_path() (+9 more)

### Community 4 - "package.json"
Cohesion: 0.09
Nodes (21): dependencies, react@18.3.1, react-dom@18.3.1, devDependencies, vite@5.4.14, @vitejs/plugin-react@4.3.4, name, private (+13 more)

### Community 6 - "faster_rcnn.yaml"
Cohesion: 0.08
Nodes (24): amp, backbone, batch, box_detections_per_img, box_nms_thresh, color_jitter, brightness, contrast (+16 more)

### Community 7 - "Parking Lot Occupancy Detection — YOLOv11 vs. Faster R-CNN"
Cohesion: 0.07
Nodes (27): 1. Clone & install Python dependencies, 2. Run the web app (API + prebuilt React UI), 3. Frontend development (hot reload), Attribution, Benchmark Results (Held-Out Test Split), Dataset Setup, Deploy with Docker Compose (recommended), Deploy with plain Docker CLI (+19 more)

### Community 8 - "inspect_model.py"
Cohesion: 0.53
Nodes (4): check_weights(), main(), stage_of(), tensor_shape()

### Community 9 - "evaluate_detector"
Cohesion: 0.15
Nodes (4): evaluate_detector(), match_predictions(), save_confusion_matrix_plot(), update_confusion_matrix()

### Community 11 - "Model & System Architecture — YOLOv11s vs. Faster R-CNN"
Cohesion: 0.08
Nodes (22): backbone, head, nc, scales, s, 1. Executive Comparison, 2. Shared Dataset & Annotation Pipeline, 3. Model 1 — YOLOv11-small (One-Stage, Anchor-Free) (+14 more)

### Community 13 - "parking-detector"
Cohesion: 0.15
Nodes (12): context, dockerfile, FASTER_RCNN_WEIGHTS, YOLO_WEIGHTS, build, container_name, environment, image (+4 more)

### Community 14 - "parse_yolo_label_file"
Cohesion: 0.22
Nodes (3): labels_dir_for(), parse_yolo_label_file(), FasterRCNNHelpersTest

### Community 16 - "resources"
Cohesion: 0.20
Nodes (9): TORCH_INDEX_URL, args, resources, build, deploy, devices, reservations, services (+1 more)

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Parking Lot Occupancy Detection — YOLOv11 vs. Faster R-CNN` connect `Parking Lot Occupancy Detection — YOLOv11 vs. Faster R-CNN` to `Model & System Architecture — YOLOv11s vs. Faster R-CNN`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `YoloParkingDataset` (e.g. with `Shared Annotation Adapter (`src/data.py`)` and `FasterRCNNHelpersTest`) actually correct?**
  _`YoloParkingDataset` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `detectors.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13538461538461538 - nodes in this community are weakly interconnected._
- **Why does `evaluate_detector()` connect `evaluate_detector` to `faster_rcnn/train.py`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Should `faster_rcnn/train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06790890269151138 - nodes in this community are weakly interconnected._
- **Should `App` be split into smaller, more focused modules?**
  _Cohesion score 0.10202020202020202 - nodes in this community are weakly interconnected._
- **Should `app.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05921325051759834 - nodes in this community are weakly interconnected._
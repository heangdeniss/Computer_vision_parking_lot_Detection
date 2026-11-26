export const HEIC = /\.(heic|heif)$/i
export const LABELS = { empty: 'Empty', occupied: 'Occupied' }
export const STRIDES = [1, 5, 15, 30, 60]
export const MODEL_MODES = [
  { value: 'both', label: 'Side by side (YOLOv11 vs Faster R-CNN)' },
  { value: 'yolo11', label: 'YOLOv11s' },
  { value: 'faster_rcnn', label: 'Faster R-CNN' },
]

// YOLOv8n-seg TFLite 모델 상수
// 텐서 shape: Input [1,640,640,3] / Output[0] [1,116,8400] / Output[1] [1,160,160,32]

export const SEG_MODEL_INPUT_SIZE = 640;
export const SEG_CONFIDENCE_THRESHOLD = 0.25;
export const SEG_IOU_THRESHOLD = 0.45;
export const SEG_MASK_THRESHOLD = 0.5;

// Output[0] shape: [1, 116, 8400]
// 116 = 4 (bbox) + 80 (COCO classes) + 32 (mask coefficients)
export const SEG_NUM_ANCHORS = 8400;
export const SEG_NUM_CLASSES = 80;
export const SEG_NUM_MASK_COEFFS = 32;
export const SEG_BBOX_DIM = 4;

// Output[1] shape: [1, 160, 160, 32]
export const SEG_PROTO_H = 160;
export const SEG_PROTO_W = 160;
export const SEG_PROTO_CH = 32;

// COCO class ID: apple = 47 (0-indexed)
export const SEG_APPLE_CLASS_ID = 47;

// 최대 탐지 수 (성능 제한)
export const SEG_MAX_DETECTIONS = 5;

// 프레임 샘플링 (N프레임당 1회 추론)
export const SEG_SAMPLE_RATE = 3;

/**
 * YOLOv8n-seg TFLite 후처리 로직 (Worklet 내 실행)
 *
 * 서버 BE/ai/services/yolov8/utils/postprocessing.py의 postprocess_seg()를
 * React Native Worklet JavaScript로 포팅.
 *
 * TFLite 출력:
 *   Output[0]: [1, 116, 8400] — pred (4 bbox + 80 classes + 32 mask_coefficients)
 *   Output[1]: [1, 160, 160, 32] — proto (mask prototypes)
 */

import {
  SEG_NUM_ANCHORS,
  SEG_NUM_CLASSES,
  SEG_NUM_MASK_COEFFS,
  SEG_BBOX_DIM,
  SEG_PROTO_H,
  SEG_PROTO_W,
  SEG_PROTO_CH,
  SEG_APPLE_CLASS_ID,
  SEG_MAX_DETECTIONS,
  SEG_CONFIDENCE_THRESHOLD,
  SEG_IOU_THRESHOLD,
  SEG_MASK_THRESHOLD,
} from '../constants/segModel';

export interface SegBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RawSegResult {
  bbox: SegBox;
  score: number;
  maskCoeffs: number[];
}

export interface SegOutputResult {
  id: number;
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
  polygon: number[][];
  score: number;
}

// ──────────────────────────────────────────────
// Worklet 유틸리티
// ──────────────────────────────────────────────

function sigmoidW(x: number): number {
  'worklet';
  return 1.0 / (1.0 + Math.exp(-x));
}

function xywh2xyxyW(
  cx: number,
  cy: number,
  w: number,
  h: number
): SegBox {
  'worklet';
  return {
    x1: cx - w / 2,
    y1: cy - h / 2,
    x2: cx + w / 2,
    y2: cy + h / 2,
  };
}

function nmsWorklet(
  boxes: SegBox[],
  scores: number[],
  iouThreshold: number,
  maxDetections: number
): number[] {
  'worklet';
  const idxs = boxes
    .map((_, i) => i)
    .sort((a, b) => scores[b] - scores[a]);
  const keep: number[] = [];

  for (let _i = 0; _i < idxs.length && keep.length < maxDetections; _i++) {
    const i = idxs[_i];
    const bi = boxes[i];
    let shouldKeep = true;

    for (let j = 0; j < keep.length; j++) {
      const bj = boxes[keep[j]];
      const xx1 = Math.max(bi.x1, bj.x1);
      const yy1 = Math.max(bi.y1, bj.y1);
      const xx2 = Math.min(bi.x2, bj.x2);
      const yy2 = Math.min(bi.y2, bj.y2);
      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const areaI = (bi.x2 - bi.x1) * (bi.y2 - bi.y1);
      const areaJ = (bj.x2 - bj.x1) * (bj.y2 - bj.y1);
      const iou = inter / (areaI + areaJ - inter + 1e-6);
      if (iou > iouThreshold) {
        shouldKeep = false;
        break;
      }
    }
    if (shouldKeep) keep.push(i);
  }
  return keep;
}

// ──────────────────────────────────────────────
// 마스크 → 폴리곤 변환 (간소화된 contour 추출)
// ──────────────────────────────────────────────

function maskToPolygonWorklet(
  mask: Float32Array,
  maskW: number,
  maskH: number,
  bbox: SegBox,
  inputSize: number,
  frameShortSide: number,
  frameCropOffsetX: number,
  frameCropOffsetY: number
): number[][] {
  'worklet';
  // 바이너리 마스크에서 경계 픽셀을 추출하여 폴리곤 근사
  // 전체 160×160 스캔 대신 bbox 영역만 스캔하여 성능 최적화
  const scaleToInput = inputSize / maskW; // 640 / 160 = 4
  const scaleToFrame = frameShortSide / inputSize;

  // bbox를 mask 좌표계로 변환
  const mx1 = Math.max(0, Math.floor(bbox.x1 / scaleToInput));
  const my1 = Math.max(0, Math.floor(bbox.y1 / scaleToInput));
  const mx2 = Math.min(maskW - 1, Math.ceil(bbox.x2 / scaleToInput));
  const my2 = Math.min(maskH - 1, Math.ceil(bbox.y2 / scaleToInput));

  // 경계 픽셀 수집 (마스크 가장자리 탐색)
  const boundaryPoints: number[][] = [];

  for (let y = my1; y <= my2; y++) {
    for (let x = mx1; x <= mx2; x++) {
      const val = mask[y * maskW + x];
      if (val <= SEG_MASK_THRESHOLD) continue;

      // 4방향 이웃 중 하나라도 마스크 밖이면 경계
      let isBoundary = false;
      if (x === mx1 || x === mx2 || y === my1 || y === my2) {
        isBoundary = true;
      } else {
        const neighbors = [
          mask[(y - 1) * maskW + x],
          mask[(y + 1) * maskW + x],
          mask[y * maskW + (x - 1)],
          mask[y * maskW + (x + 1)],
        ];
        for (let n = 0; n < 4; n++) {
          if (neighbors[n] <= SEG_MASK_THRESHOLD) {
            isBoundary = true;
            break;
          }
        }
      }

      if (isBoundary) {
        // mask 좌표 → 원본 프레임 좌표
        const frameX = x * scaleToInput * scaleToFrame + frameCropOffsetX;
        const frameY = y * scaleToInput * scaleToFrame + frameCropOffsetY;
        boundaryPoints.push([frameX, frameY]);
      }
    }
  }

  if (boundaryPoints.length < 3) return [];

  // 중심점 기준 각도 정렬 (시계방향 폴리곤)
  let cx = 0,
    cy = 0;
  for (let i = 0; i < boundaryPoints.length; i++) {
    cx += boundaryPoints[i][0];
    cy += boundaryPoints[i][1];
  }
  cx /= boundaryPoints.length;
  cy /= boundaryPoints.length;

  boundaryPoints.sort((a, b) => {
    const angleA = Math.atan2(a[1] - cy, a[0] - cx);
    const angleB = Math.atan2(b[1] - cy, b[0] - cx);
    return angleA - angleB;
  });

  // 포인트 수 제한 (성능: 최대 64개로 다운샘플링)
  const maxPoints = 64;
  if (boundaryPoints.length <= maxPoints) return boundaryPoints;

  const step = boundaryPoints.length / maxPoints;
  const sampled: number[][] = [];
  for (let i = 0; i < maxPoints; i++) {
    sampled.push(boundaryPoints[Math.floor(i * step)]);
  }
  return sampled;
}

// ──────────────────────────────────────────────
// 메인 후처리 함수 (Worklet)
// ──────────────────────────────────────────────

export function postprocessSegWorklet(
  predRaw: Float32Array, // Output[0]: [1, 116, 8400] flattened
  protoRaw: Float32Array, // Output[1]: [1, 160, 160, 32] flattened
  inputSize: number,
  frameWidth: number,
  frameHeight: number
): SegOutputResult[] {
  'worklet';

  // Output[0] shape: [1, 116, 8400] → 116 rows × 8400 columns
  // Row layout: [0..3]=bbox(cx,cy,w,h), [4..83]=class_probs, [84..115]=mask_coeffs
  const ROWS = SEG_BBOX_DIM + SEG_NUM_CLASSES + SEG_NUM_MASK_COEFFS; // 116
  const COLS = SEG_NUM_ANCHORS; // 8400

  // 크롭 파라미터 (전처리에서 중앙 정사각형 크롭)
  const shortSide = Math.min(frameWidth, frameHeight);
  const cropOffsetX = (frameWidth - shortSide) / 2;
  const cropOffsetY = (frameHeight - shortSide) / 2;
  const scaleToFrame = shortSide / inputSize;

  // 1. 사과 클래스 필터링 + score 계산
  const candidates: RawSegResult[] = [];

  for (let col = 0; col < COLS; col++) {
    // 사과 클래스 확률 (class index 47)
    const classOffset = SEG_BBOX_DIM + SEG_APPLE_CLASS_ID;
    const appleScore = predRaw[classOffset * COLS + col];

    if (appleScore < SEG_CONFIDENCE_THRESHOLD) continue;

    // bbox (cx, cy, w, h) — 이미 픽셀 단위 (640×640 기준)
    const cx = predRaw[0 * COLS + col];
    const cy = predRaw[1 * COLS + col];
    const w = predRaw[2 * COLS + col];
    const h = predRaw[3 * COLS + col];

    const box = xywh2xyxyW(cx, cy, w, h);

    // 유효성 검사
    if (box.x2 <= box.x1 || box.y2 <= box.y1) continue;
    if (box.x1 < -10 || box.y1 < -10 || box.x2 > inputSize + 10 || box.y2 > inputSize + 10)
      continue;

    // mask coefficients 추출
    const maskCoeffs: number[] = [];
    const maskOffset = SEG_BBOX_DIM + SEG_NUM_CLASSES;
    for (let m = 0; m < SEG_NUM_MASK_COEFFS; m++) {
      maskCoeffs.push(predRaw[(maskOffset + m) * COLS + col]);
    }

    candidates.push({ bbox: box, score: appleScore, maskCoeffs });
  }

  if (candidates.length === 0) return [];

  // 2. NMS
  const boxes = candidates.map((c) => c.bbox);
  const scores = candidates.map((c) => c.score);
  const keepIdx = nmsWorklet(boxes, scores, SEG_IOU_THRESHOLD, SEG_MAX_DETECTIONS);

  // 3. Proto를 [PROTO_CH, PROTO_H*PROTO_W] 로 재배열
  // protoRaw shape: [1, 160, 160, 32] (NHWC) → [32, 25600]
  const protoFlat = new Float32Array(SEG_PROTO_CH * SEG_PROTO_H * SEG_PROTO_W);
  for (let c = 0; c < SEG_PROTO_CH; c++) {
    for (let y = 0; y < SEG_PROTO_H; y++) {
      for (let x = 0; x < SEG_PROTO_W; x++) {
        // NHWC: [0, y, x, c] → index = y*W*C + x*C + c
        const srcIdx = y * SEG_PROTO_W * SEG_PROTO_CH + x * SEG_PROTO_CH + c;
        // target: [c, y*W+x]
        const dstIdx = c * (SEG_PROTO_H * SEG_PROTO_W) + y * SEG_PROTO_W + x;
        protoFlat[dstIdx] = protoRaw[srcIdx];
      }
    }
  }

  // 4. 각 탐지에 대해 마스크 생성 + 폴리곤 추출
  const results: SegOutputResult[] = [];
  const protoPixels = SEG_PROTO_H * SEG_PROTO_W; // 25600

  for (let ki = 0; ki < keepIdx.length; ki++) {
    const det = candidates[keepIdx[ki]];

    // mask = coeffs[1×32] × proto[32×25600] → [25600]
    const mask = new Float32Array(protoPixels);
    for (let px = 0; px < protoPixels; px++) {
      let sum = 0;
      for (let c = 0; c < SEG_NUM_MASK_COEFFS; c++) {
        sum += det.maskCoeffs[c] * protoFlat[c * protoPixels + px];
      }
      mask[px] = sigmoidW(sum);
    }

    // 폴리곤 추출
    const polygon = maskToPolygonWorklet(
      mask,
      SEG_PROTO_W,
      SEG_PROTO_H,
      det.bbox,
      inputSize,
      shortSide,
      cropOffsetX,
      cropOffsetY
    );

    if (polygon.length < 3) continue;

    // bbox를 프레임 좌표로 변환
    const xmin = Math.round(det.bbox.x1 * scaleToFrame + cropOffsetX);
    const ymin = Math.round(det.bbox.y1 * scaleToFrame + cropOffsetY);
    const xmax = Math.round(det.bbox.x2 * scaleToFrame + cropOffsetX);
    const ymax = Math.round(det.bbox.y2 * scaleToFrame + cropOffsetY);

    results.push({
      id: ki,
      bbox: { xmin, ymin, xmax, ymax },
      polygon,
      score: det.score,
    });
  }

  return results;
}

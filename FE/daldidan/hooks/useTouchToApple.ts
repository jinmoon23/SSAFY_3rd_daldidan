/**
 * useTouchToApple — 터치 좌표 → 사과 폴리곤 매핑 훅 (Phase 2)
 *
 * 화면 터치 좌표를 프레임 좌표로 역변환한 뒤,
 * Ray Casting 알고리즘으로 어떤 사과 폴리곤 내부인지 판정.
 * Fallback: bbox 내부 체크.
 */

import { useCallback } from 'react';
import { SegmentationResult } from './types/objectDetection';

interface UseTouchToAppleParams {
  segmentations: SegmentationResult[];
  screenSize: { width: number; height: number };
  frameSize: { width: number; height: number };
}

/**
 * 화면 좌표(Portrait) → 프레임 좌표(Landscape) 역변환
 *
 * RealtimeSegOverlay의 transformToScreen의 역함수:
 *   forward:  rotatedX = frameH - frameY,  rotatedY = frameX
 *             screenX  = rotatedX * scale + offsetX
 *             screenY  = rotatedY * scale
 *
 *   inverse:  rotatedX = (screenX - offsetX) / scale
 *             rotatedY = screenY / scale
 *             frameX   = rotatedY
 *             frameY   = frameH - rotatedX
 */
function screenToFrame(
  screenX: number,
  screenY: number,
  frameW: number,
  frameH: number,
  screenW: number,
  screenH: number
): { x: number; y: number } {
  const rotatedW = frameH;
  const rotatedH = frameW;
  const scale = screenH / rotatedH;
  const scaledW = rotatedW * scale;
  const offsetX = (screenW - scaledW) / 2;

  const rotatedX = (screenX - offsetX) / scale;
  const rotatedY = screenY / scale;

  // 역회전: rotatedX = frameH - frameY → frameY = frameH - rotatedX
  //         rotatedY = frameX          → frameX = rotatedY
  const frameX = rotatedY;
  const frameY = frameH - rotatedX;

  return { x: frameX, y: frameY };
}

/**
 * Ray Casting 알고리즘으로 점이 폴리곤 내부인지 판정
 * polygon: [[x,y], [x,y], ...] (프레임 좌표)
 */
function pointInPolygon(
  px: number,
  py: number,
  polygon: number[][]
): boolean {
  let inside = false;
  const n = polygon.length;

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i][0],
      yi = polygon[i][1];
    const xj = polygon[j][0],
      yj = polygon[j][1];

    const intersect =
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }

  return inside;
}

/**
 * Fallback: 점이 bbox 내부인지 판정
 */
function pointInBbox(
  px: number,
  py: number,
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number }
): boolean {
  return px >= bbox.xmin && px <= bbox.xmax && py >= bbox.ymin && py <= bbox.ymax;
}

export function useTouchToApple({
  segmentations,
  screenSize,
  frameSize,
}: UseTouchToAppleParams) {
  /**
   * 화면 터치 좌표 → 매칭된 사과 id 반환 (없으면 null)
   * 우선순위: polygon 내부 → bbox 내부 (fallback)
   */
  const findAppleAtTouch = useCallback(
    (screenX: number, screenY: number): number | null => {
      if (
        segmentations.length === 0 ||
        screenSize.width === 0 ||
        screenSize.height === 0
      ) {
        return null;
      }

      const frame = screenToFrame(
        screenX,
        screenY,
        frameSize.width,
        frameSize.height,
        screenSize.width,
        screenSize.height
      );

      // 1차: 폴리곤 내부 판정
      for (const seg of segmentations) {
        if (seg.polygon.length >= 3 && pointInPolygon(frame.x, frame.y, seg.polygon)) {
          return seg.id;
        }
      }

      // 2차 Fallback: bbox 내부 판정
      for (const seg of segmentations) {
        if (pointInBbox(frame.x, frame.y, seg.bbox)) {
          return seg.id;
        }
      }

      return null;
    },
    [segmentations, screenSize, frameSize]
  );

  return { findAppleAtTouch };
}

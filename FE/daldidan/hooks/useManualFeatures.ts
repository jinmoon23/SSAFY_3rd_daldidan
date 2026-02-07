/**
 * useManualFeatures — 수동 특징 추출 (Worklet 호환)
 * Phase 3 Step 3-3
 *
 * RGB 픽셀 데이터에서 6개 수동 특징 추출:
 * [Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow]
 *
 * 서버의 extract_features.py와 동일한 로직
 */

import scalerValues from '../constants/scalerValues.json';

// Scaler 상수 (worklet에서 접근 가능하도록 모듈 레벨 추출)
const SCALER_MEAN = scalerValues.mean;
const SCALER_SCALE = scalerValues.scale;

/**
 * Worklet: RGB uint8 픽셀에서 6개 수동 특징 추출
 *
 * 서버 extract_features(image, mask)와 동일:
 *   - ROI는 이미 bbox로 크롭되어 64×64로 리사이즈된 상태
 *   - RGB 채널 순서 (카메라 프레임 기준)
 *
 * OpenCV COLOR_BGR2YCrCb 주의:
 *   서버 코드에서 변수명이 뒤바뀌어 있음 (Cb, Cr = YCrCb[:,:,1], YCrCb[:,:,2])
 *   YCrCb 출력 채널: [Y, Cr, Cb] → 서버의 "Cb" = 실제 Cr, "Cr" = 실제 Cb
 *   모델 호환성을 위해 동일하게 구현
 */
export function extractManualFeaturesWorklet(
  rgbData: any,
  width: number,
  height: number
): number[] {
  'worklet';

  const pixels = new Uint8Array(rgbData);
  const totalPixels = width * height;

  let sumRn = 0;
  let sumC = 0;
  // 서버의 swapped naming 그대로 유지
  let sumServerCb = 0; // 실제로는 Cr 채널 값
  let sumServerCr = 0; // 실제로는 Cb 채널 값

  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 3;
    const R = pixels[idx];
    const G = pixels[idx + 1];
    const B = pixels[idx + 2];

    // Rn = R / (R + G + B + eps)
    const sumRGB = R + G + B + 0.00001;
    sumRn += R / sumRGB;

    // C = 1 - R / 255
    sumC += 1 - R / 255;

    // YCbCr (OpenCV COLOR_BGR2YCrCb)
    // 출력: [Y, Cr, Cb]
    const Y = 0.299 * R + 0.587 * G + 0.114 * B;
    const actualCr = (R - Y) * 0.713 + 128; // YCrCb channel 1
    const actualCb = (B - Y) * 0.564 + 128; // YCrCb channel 2

    // 서버의 Cb, Cr = YCrCb[:,:,1], YCrCb[:,:,2]
    sumServerCb += actualCr; // 서버 "Cb" = 실제 Cr
    sumServerCr += actualCb; // 서버 "Cr" = 실제 Cb
  }

  const Rn = sumRn / totalPixels;
  const C = sumC / totalPixels;
  const cbMean = sumServerCb / totalPixels;
  const crMean = sumServerCr / totalPixels;
  const ycbcrDiff = cbMean - crMean;
  const ycbcrNorm = cbMean / (cbMean + crMean + 0.00001);

  // cat02_first: 0.0 (서버 추론 코드에서도 0.0 고정)
  const cat02First = 0.0;

  // cluster_shadow: GLCM contrast (distance=1, angle=0, levels=32, symmetric)
  const levels = 32;
  const grayQ: number[] = [];
  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 3;
    const grayVal =
      0.299 * pixels[idx] + 0.587 * pixels[idx + 1] + 0.114 * pixels[idx + 2];
    grayQ.push(Math.min(Math.floor(grayVal / 8), levels - 1));
  }

  // GLCM 행렬 (32×32, symmetric, distance=1, angle=0 → horizontal)
  const glcm: number[] = new Array(levels * levels).fill(0);
  let pairCount = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      const gi = grayQ[y * width + x];
      const gj = grayQ[y * width + x + 1];
      glcm[gi * levels + gj]++;
      glcm[gj * levels + gi]++;
      pairCount += 2;
    }
  }

  // Contrast = Σ (i-j)² × P(i,j)
  let contrast = 0;
  if (pairCount > 0) {
    for (let i = 0; i < levels; i++) {
      for (let j = 0; j < levels; j++) {
        const p = glcm[i * levels + j] / pairCount;
        contrast += (i - j) * (i - j) * p;
      }
    }
  }

  return [Rn, C, ycbcrDiff, ycbcrNorm, cat02First, contrast];
}

/**
 * JS: 수동 특징을 StandardScaler로 정규화
 * formula: scaled = (x - mean) / scale
 */
export function scaleManualFeatures(features: number[]): number[] {
  return features.map((x, i) => (x - SCALER_MEAN[i]) / SCALER_SCALE[i]);
}

/**
 * Worklet: 수동 특징을 StandardScaler로 정규화
 */
export function scaleManualFeaturesWorklet(features: number[]): number[] {
  'worklet';
  const mean = SCALER_MEAN;
  const scale = SCALER_SCALE;
  const result: number[] = [];
  for (let i = 0; i < features.length; i++) {
    result.push((features[i] - mean[i]) / scale[i]);
  }
  return result;
}

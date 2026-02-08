/**
 * useSweetnessPredictor — 온디바이스 당도 예측 파이프라인
 * Phase 3 Step 3-2
 *
 * EfficientNet-B0 TFLite → 1280 CNN features
 * + Manual Features (6) → StandardScaler
 * → MLP (1286 → 128 → 1) → 당도 (Brix)
 *
 * 사용법:
 *   const sweetness = useSweetnessPredictor();
 *   // useSegmentation에 sweetness 설정 전달
 *   // 터치 시: sweetness.requestPrediction(appleId, bbox)
 *   // 결과: sweetness.predictionResult
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import { Worklets } from 'react-native-worklets-core';
import { Asset } from 'expo-asset';
import { scaleManualFeatures } from './useManualFeatures';

interface MlpWeights {
  fc1: { weight: number[][]; bias: number[] };
  fc2: { weight: number[][]; bias: number[] };
}

// ImageNet 정규화 상수 (EfficientNet-B0 입력)
export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

// EfficientNet 입력 크기
export const EFFICIENTNET_INPUT_SIZE = 224;

// Manual features용 크롭 크기
export const MANUAL_FEATURE_CROP_SIZE = 64;

export interface CropRequest {
  appleId: number;
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
}

export interface PredictionResult {
  appleId: number;
  sweetness: number;
}

// Fingerprint 매칭 결과 (자동 당도 복원)
export interface FingerprintMatchResult {
  appleId: number;
  sweetness: number;
}

// 멀티프레임 앙상블 설정
const ENSEMBLE_FRAME_COUNT = 5;

// Fingerprint 매칭 상수
const FINGERPRINT_MATCH_THRESHOLD = 0.82;
const FINGERPRINT_SPATIAL_WEIGHT = 0.2;
const FINGERPRINT_CNN_WEIGHT = 0.8;
const MAX_FINGERPRINTS = 20;

interface EnsembleState {
  appleId: number;
  bbox: CropRequest['bbox'];
  predictions: number[];
  lastCnnFeatures?: number[];
}

interface AppleFingerprint {
  cnnFeatures: number[];
  normalizedCenter: { x: number; y: number };
  sweetness: number;
  timestamp: number;
}

export function useSweetnessPredictor() {
  const modelRef = useRef<TensorflowModel | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const weightsRef = useRef<MlpWeights | null>(null);

  // 멀티프레임 앙상블 상태 (사과별 독립 추적)
  const ensembleMapRef = useRef<Map<number, EnsembleState>>(new Map());

  // Fingerprint 캐시 (영구 저장)
  const fingerprintCacheRef = useRef<AppleFingerprint[]>([]);

  // Fingerprint 매칭 결과
  const [fingerprintMatch, setFingerprintMatch] =
    useState<FingerprintMatchResult | null>(null);

  // SharedValue: 당도 예측용 크롭 큐 (JSON 배열)
  const cropQueue = useRef(Worklets.createSharedValue<string>('[]')).current;

  // SharedValue: fingerprint 추출용 크롭 큐 (JSON 배열)
  const fingerprintQueue = useRef(Worklets.createSharedValue<string>('[]')).current;

  // ─────────────────────────────────────────
  // MLP 추론 (JS 스레드)
  // ─────────────────────────────────────────
  const mlpPredict = useCallback(
    (cnnFeatures: number[], scaledManualFeatures: number[]): number | null => {
      const w = weightsRef.current;
      if (!w) return null;

      const input = [...cnnFeatures, ...scaledManualFeatures]; // [1286]

      // FC1: [128 × 1286] + bias → ReLU
      const hidden = new Array(128);
      for (let i = 0; i < 128; i++) {
        let sum = w.fc1.bias[i];
        const row = w.fc1.weight[i];
        for (let j = 0; j < 1286; j++) {
          sum += row[j] * input[j];
        }
        hidden[i] = Math.max(0, sum); // ReLU
      }

      // FC2: [1 × 128] + bias → 당도
      let output = w.fc2.bias[0];
      const row2 = w.fc2.weight[0];
      for (let i = 0; i < 128; i++) {
        output += row2[i] * hidden[i];
      }

      return output;
    },
    []
  );

  // ─────────────────────────────────────────
  // 코사인 유사도 계산
  // ─────────────────────────────────────────
  const cosineSimilarity = (a: number[], b: number[]): number => {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  };

  // ─────────────────────────────────────────
  // Fingerprint 저장
  // ─────────────────────────────────────────
  const storeFingerprint = (
    cnnFeatures: number[],
    bbox: CropRequest['bbox'],
    sweetness: number,
    frameW: number,
    frameH: number
  ) => {
    const cache = fingerprintCacheRef.current;
    const normalizedCenter = {
      x: ((bbox.xmin + bbox.xmax) / 2) / (frameW || 1),
      y: ((bbox.ymin + bbox.ymax) / 2) / (frameH || 1),
    };
    cache.push({ cnnFeatures, normalizedCenter, sweetness, timestamp: Date.now() });
    // 캐시 크기 제한
    if (cache.length > MAX_FINGERPRINTS) {
      cache.splice(0, cache.length - MAX_FINGERPRINTS);
    }
    console.log(`[Fingerprint] Stored. Cache size: ${cache.length}`);
  };

  // ─────────────────────────────────────────
  // Fingerprint 매칭 (새 사과 → 캐시 비교)
  // ─────────────────────────────────────────
  const matchFingerprint = (
    cnnFeatures: number[],
    normalizedCenter: { x: number; y: number }
  ): AppleFingerprint | null => {
    const cache = fingerprintCacheRef.current;
    if (cache.length === 0) return null;

    let bestScore = 0;
    let bestMatch: AppleFingerprint | null = null;

    for (const fp of cache) {
      const cnnSim = cosineSimilarity(cnnFeatures, fp.cnnFeatures);
      const dx = normalizedCenter.x - fp.normalizedCenter.x;
      const dy = normalizedCenter.y - fp.normalizedCenter.y;
      const spatialSim = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy));
      const score = FINGERPRINT_CNN_WEIGHT * cnnSim + FINGERPRINT_SPATIAL_WEIGHT * spatialSim;

      if (score > bestScore) {
        bestScore = score;
        bestMatch = fp;
      }
    }

    if (bestScore >= FINGERPRINT_MATCH_THRESHOLD && bestMatch) {
      console.log(`[Fingerprint] Match found! score=${bestScore.toFixed(3)}, sweetness=${bestMatch.sweetness.toFixed(2)}`);
      return bestMatch;
    }
    return null;
  };

  // ─────────────────────────────────────────
  // 중앙값 계산
  // ─────────────────────────────────────────
  const computeMedian = (values: number[]): number => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  };

  // ─────────────────────────────────────────
  // Worklet → JS: CNN features + manual features 수신
  // 멀티프레임 앙상블: N회 수집 후 중앙값으로 확정
  // ─────────────────────────────────────────
  const handleFeaturesFromWorklet = useRef(
    Worklets.createRunOnJS(
      (appleId: number, cnnFeatures: number[], manualFeatures: number[], frameW: number, frameH: number) => {
        try {
          const ensemble = ensembleMapRef.current.get(appleId);
          if (!ensemble) return;

          // 1. Scale manual features
          const scaled = scaleManualFeatures(manualFeatures);

          // 2. MLP 추론
          const sweetness = mlpPredict(cnnFeatures, scaled);
          if (sweetness === null) {
            console.warn('[Sweetness] MLP weights not loaded yet');
            return;
          }

          // 3. 앙상블 버퍼에 추가 + CNN features 보관
          ensemble.predictions.push(sweetness);
          ensemble.lastCnnFeatures = cnnFeatures;
          console.log(
            `[Sweetness] Apple #${appleId} frame ${ensemble.predictions.length}/${ENSEMBLE_FRAME_COUNT}: ${sweetness.toFixed(2)} Brix`
          );

          // 4. 충분히 모였으면 중앙값으로 확정 + fingerprint 저장
          if (ensemble.predictions.length >= ENSEMBLE_FRAME_COUNT) {
            const median = computeMedian(ensemble.predictions);
            console.log(
              `[Sweetness] Apple #${appleId} FINAL (median of ${ENSEMBLE_FRAME_COUNT}): ${median.toFixed(2)} Brix`
            );
            // Fingerprint 캐시에 저장
            if (ensemble.lastCnnFeatures) {
              storeFingerprint(
                ensemble.lastCnnFeatures,
                ensemble.bbox,
                median,
                frameW,
                frameH
              );
            }
            ensembleMapRef.current.delete(appleId);
            setPredictionResult({ appleId, sweetness: median });
          } else {
            // 5. 아직 부족하면 큐에 다음 크롭 요청 추가
            try {
              const queue: CropRequest[] = JSON.parse(cropQueue.value);
              queue.push({ appleId: ensemble.appleId, bbox: ensemble.bbox });
              cropQueue.value = JSON.stringify(queue);
            } catch {
              cropQueue.value = JSON.stringify([{ appleId: ensemble.appleId, bbox: ensemble.bbox }]);
            }
          }
        } catch (error: any) {
          console.error(
            `[Sweetness] MLP prediction error for apple #${appleId}:`,
            error.message
          );
        }
      }
    )
  ).current;

  // ─────────────────────────────────────────
  // Worklet → JS: Fingerprint CNN features 수신 (자동 re-ID)
  // ─────────────────────────────────────────
  const handleFingerprintFromWorklet = useRef(
    Worklets.createRunOnJS(
      (appleId: number, cnnFeatures: number[], normalizedCx: number, normalizedCy: number) => {
        try {
          const match = matchFingerprint(cnnFeatures, { x: normalizedCx, y: normalizedCy });
          if (match) {
            console.log(`[Fingerprint] Apple #${appleId} → restored ${match.sweetness.toFixed(2)} Brix`);
            setFingerprintMatch({ appleId, sweetness: match.sweetness });
          }
        } catch (error: any) {
          console.error(`[Fingerprint] Match error: ${error.message}`);
        }
      }
    )
  ).current;

  // ─────────────────────────────────────────
  // JS → Worklet: 크롭 요청 (터치 시 호출)
  // 멀티프레임 앙상블 시작 — 사과별 독립
  // ─────────────────────────────────────────
  const requestPrediction = useCallback(
    (appleId: number, bbox: CropRequest['bbox']) => {
      if (!isModelLoaded) {
        console.warn('[Sweetness] Model not loaded yet');
        return;
      }
      // 앙상블 Map에 추가 (기존 엔트리가 있으면 초기화)
      ensembleMapRef.current.set(appleId, { appleId, bbox, predictions: [] });
      // 큐에 크롭 요청 추가
      try {
        const queue: CropRequest[] = JSON.parse(cropQueue.value);
        queue.push({ appleId, bbox });
        cropQueue.value = JSON.stringify(queue);
      } catch {
        cropQueue.value = JSON.stringify([{ appleId, bbox }]);
      }
    },
    [isModelLoaded]
  );

  // ─────────────────────────────────────────
  // JS → Worklet: Fingerprint 크롭 요청 (자동 re-ID용)
  // ─────────────────────────────────────────
  const requestFingerprint = useCallback(
    (appleId: number, bbox: CropRequest['bbox']) => {
      if (!isModelLoaded) return;
      if (fingerprintCacheRef.current.length === 0) return;
      // 큐에 fingerprint 요청 추가
      try {
        const queue: CropRequest[] = JSON.parse(fingerprintQueue.value);
        queue.push({ appleId, bbox });
        fingerprintQueue.value = JSON.stringify(queue);
      } catch {
        fingerprintQueue.value = JSON.stringify([{ appleId, bbox }]);
      }
    },
    [isModelLoaded]
  );

  // ─────────────────────────────────────────
  // EfficientNet 모델 + MLP 가중치 로딩 (비동기)
  // ─────────────────────────────────────────
  useEffect(() => {
    const loadAll = async () => {
      // 1. MLP 가중치 비동기 로드 (3.6MB — JS 번들에 인라인하지 않음)
      try {
        console.log('[Sweetness] Loading MLP weights...');
        const asset = Asset.fromModule(
          require('../assets/mlpWeights.bin')
        );
        await asset.downloadAsync();
        const response = await fetch(asset.localUri!);
        const weights: MlpWeights = await response.json();
        weightsRef.current = weights;
        console.log('[Sweetness] MLP weights loaded');
      } catch (error: any) {
        console.error('[Sweetness] MLP weights loading failed:', error.message);
      }

      // 2. EfficientNet TFLite 모델 로드
      try {
        console.log('[Sweetness] Loading EfficientNet-B0 TFLite...');
        const model = await loadTensorflowModel(
          require('../assets/efficientnet_b0_apple.tflite'),
          'gpu' as TensorflowModelDelegate
        );
        console.log('[Sweetness] EfficientNet model loaded successfully');
        modelRef.current = model;
        setIsModelLoaded(true);
      } catch (error: any) {
        console.error(
          '[Sweetness] EfficientNet loading failed:',
          error.message
        );
      }
    };
    loadAll();

    return () => {
      modelRef.current = null;
      weightsRef.current = null;
      setIsModelLoaded(false);
    };
  }, []);

  return {
    // Refs (frame processor에서 사용)
    sweetnessModelRef: modelRef,
    cropQueue,
    fingerprintQueue,
    handleFeaturesFromWorklet,
    handleFingerprintFromWorklet,

    // JS API
    requestPrediction,
    requestFingerprint,
    predictionResult,
    fingerprintMatch,
    isModelLoaded,
    mlpPredict,
  };
}

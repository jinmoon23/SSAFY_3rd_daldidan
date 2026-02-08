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

// 멀티프레임 앙상블 설정
const ENSEMBLE_FRAME_COUNT = 5;

interface EnsembleState {
  appleId: number;
  bbox: CropRequest['bbox'];
  predictions: number[];
}

export function useSweetnessPredictor() {
  const modelRef = useRef<TensorflowModel | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const weightsRef = useRef<MlpWeights | null>(null);

  // 멀티프레임 앙상블 상태
  const ensembleRef = useRef<EnsembleState | null>(null);

  // SharedValue: 크롭 요청 (JSON 문자열, worklet에서 파싱)
  const cropRequest = useRef(Worklets.createSharedValue<string | null>(null)).current;

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
      (appleId: number, cnnFeatures: number[], manualFeatures: number[]) => {
        try {
          const ensemble = ensembleRef.current;
          if (!ensemble || ensemble.appleId !== appleId) return;

          // 1. Scale manual features
          const scaled = scaleManualFeatures(manualFeatures);

          // 2. MLP 추론
          const sweetness = mlpPredict(cnnFeatures, scaled);
          if (sweetness === null) {
            console.warn('[Sweetness] MLP weights not loaded yet');
            return;
          }

          // 3. 앙상블 버퍼에 추가
          ensemble.predictions.push(sweetness);
          console.log(
            `[Sweetness] Apple #${appleId} frame ${ensemble.predictions.length}/${ENSEMBLE_FRAME_COUNT}: ${sweetness.toFixed(2)} Brix`
          );

          // 4. 충분히 모였으면 중앙값으로 확정
          if (ensemble.predictions.length >= ENSEMBLE_FRAME_COUNT) {
            const median = computeMedian(ensemble.predictions);
            console.log(
              `[Sweetness] Apple #${appleId} FINAL (median of ${ENSEMBLE_FRAME_COUNT}): ${median.toFixed(2)} Brix`
            );
            ensembleRef.current = null;
            setPredictionResult({ appleId, sweetness: median });
          } else {
            // 5. 아직 부족하면 다음 프레임에 다시 크롭 요청
            cropRequest.value = JSON.stringify({
              appleId: ensemble.appleId,
              bbox: ensemble.bbox,
            });
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
  // JS → Worklet: 크롭 요청 (터치 시 호출)
  // 멀티프레임 앙상블 시작
  // ─────────────────────────────────────────
  const requestPrediction = useCallback(
    (appleId: number, bbox: CropRequest['bbox']) => {
      if (!isModelLoaded) {
        console.warn('[Sweetness] Model not loaded yet');
        return;
      }
      // 앙상블 상태 초기화
      ensembleRef.current = { appleId, bbox, predictions: [] };
      cropRequest.value = JSON.stringify({ appleId, bbox });
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
    cropRequest,
    handleFeaturesFromWorklet,

    // JS API
    requestPrediction,
    predictionResult,
    isModelLoaded,
    mlpPredict,
  };
}

/**
 * useSegmentation — YOLOv8n-seg 온디바이스 실시간 세그멘테이션 훅
 *
 * 카메라 프레임을 Worklet에서 처리하여 사과의 세그멘테이션 마스크를 실시간 추출.
 * Phase 3: 선택적으로 EfficientNet-B0 크롭 캡처 + 수동 특징 추출 지원.
 */

import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import { Camera, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets, ISharedValue } from 'react-native-worklets-core';
import { useImageProcessing } from './useImageProcessing';
import { postprocessSegWorklet, SegOutputResult } from './useSegPostprocessing';
import { SegmentationResult } from './types/objectDetection';
import {
  SEG_MODEL_INPUT_SIZE,
  SEG_SAMPLE_RATE,
} from '../constants/segModel';
import { extractManualFeaturesWorklet } from './useManualFeatures';
import {
  CropRequest,
  EFFICIENTNET_INPUT_SIZE,
  MANUAL_FEATURE_CROP_SIZE,
  IMAGENET_MEAN,
  IMAGENET_STD,
} from './useSweetnessPredictor';

export interface SweetnessConfig {
  sweetnessModelRef: React.RefObject<TensorflowModel | null>;
  cropRequest: ISharedValue<string | null>;
  handleFeaturesFromWorklet: (
    appleId: number,
    cnnFeatures: number[],
    manualFeatures: number[]
  ) => void;
}

export function useSegmentation(
  format: any,
  sweetnessConfig?: SweetnessConfig
) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const cameraRef = useRef<Camera>(null);
  const frameCount = useRef(Worklets.createSharedValue(0)).current;
  const [segmentations, setSegmentations] = useState<SegmentationResult[]>([]);
  const [hasPermission, setHasPermission] = useState(false);

  // Phase 3: sweetnessConfig를 ref에 저장 → frame processor 재생성 방지
  const sweetnessConfigRef = useRef(sweetnessConfig);
  sweetnessConfigRef.current = sweetnessConfig;

  const { preprocessFrameForSeg, cropAndResize, cropAndResizeUint8, logWorklet } =
    useImageProcessing();

  // Worklet → JS 스레드로 세그멘테이션 결과 전달
  const updateSegmentationsWorklet = useRef(
    Worklets.createRunOnJS((data: SegmentationResult[]) => {
      setSegmentations(data);
    })
  ).current;

  // Worklet 내에서 추론 + 후처리
  const processSegmentationInWorklet = (
    frame: any,
    model: TensorflowModel
  ): SegmentationResult[] => {
    'worklet';
    try {
      const resized = preprocessFrameForSeg(frame, SEG_MODEL_INPUT_SIZE);
      const outputs = model.runSync([resized]);

      // Output[0]: [1, 116, 8400], Output[1]: [1, 160, 160, 32]
      const predRaw = outputs[0] as Float32Array;
      const protoRaw = outputs[1] as Float32Array;

      const results = postprocessSegWorklet(
        predRaw,
        protoRaw,
        SEG_MODEL_INPUT_SIZE,
        frame.width,
        frame.height
      );

      // SegOutputResult → SegmentationResult 변환
      const segResults: SegmentationResult[] = [];
      for (let i = 0; i < results.length; i++) {
        segResults.push({
          id: results[i].id,
          bbox: results[i].bbox,
          polygon: results[i].polygon,
          score: results[i].score,
        });
      }

      return segResults;
    } catch (error) {
      logWorklet(`[Worklet] Segmentation error: ${error}`);
      return [];
    }
  };

  // FrameProcessor — 매 N프레임마다 세그멘테이션 추론 + 온디맨드 당도 크롭
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!modelRef.current) return;

      // ── Phase 3: 당도 크롭 요청 처리 (매 프레임 체크) ──
      const sc = sweetnessConfigRef.current;
      if (sc) {
        const requestStr = sc.cropRequest.value;
        if (requestStr && sc.sweetnessModelRef.current) {
          sc.cropRequest.value = null; // 즉시 클리어

          try {
            const request = JSON.parse(requestStr) as CropRequest;
            const { appleId, bbox } = request;
            const cropX = Math.max(0, Math.floor(bbox.xmin));
            const cropY = Math.max(0, Math.floor(bbox.ymin));
            const cropW = Math.min(
              Math.floor(bbox.xmax - bbox.xmin),
              frame.width - cropX
            );
            const cropH = Math.min(
              Math.floor(bbox.ymax - bbox.ymin),
              frame.height - cropY
            );

            if (cropW >= 10 && cropH >= 10) {
              // 1. EfficientNet 입력: 224×224 float32 (0~1)
              const cnnInput = cropAndResize(
                frame,
                cropX,
                cropY,
                cropW,
                cropH,
                EFFICIENTNET_INPUT_SIZE
              );

              if (cnnInput) {
                // 2. ImageNet 정규화: (pixel - mean) / std
                const floatView = new Float32Array(cnnInput);
                const px = EFFICIENTNET_INPUT_SIZE * EFFICIENTNET_INPUT_SIZE;
                for (let i = 0; i < px; i++) {
                  const b = i * 3;
                  floatView[b] = (floatView[b] - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                  floatView[b + 1] =
                    (floatView[b + 1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                  floatView[b + 2] =
                    (floatView[b + 2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                }

                // 3. EfficientNet 추론 → 1280 features
                const cnnOutputs =
                  sc.sweetnessModelRef.current!.runSync([cnnInput]);
                const cnnFeaturesRaw = cnnOutputs[0] as Float32Array;
                const cnnFeatures: number[] = [];
                for (let i = 0; i < cnnFeaturesRaw.length; i++) {
                  cnnFeatures.push(cnnFeaturesRaw[i]);
                }

                // 4. Manual features 용 64×64 uint8 크롭
                const manualInput = cropAndResizeUint8(
                  frame,
                  cropX,
                  cropY,
                  cropW,
                  cropH,
                  MANUAL_FEATURE_CROP_SIZE
                );

                if (manualInput) {
                  // 5. Manual features 추출
                  const manualFeatures = extractManualFeaturesWorklet(
                    manualInput,
                    MANUAL_FEATURE_CROP_SIZE,
                    MANUAL_FEATURE_CROP_SIZE
                  );

                  // 6. JS로 전송 → MLP 추론
                  sc.handleFeaturesFromWorklet(
                    appleId,
                    cnnFeatures,
                    manualFeatures
                  );
                }
              }
            }
          } catch (error) {
            logWorklet(`[Worklet] Sweetness crop error: ${error}`);
          }
        }
      }

      // ── 기존: 세그멘테이션 추론 (매 N프레임) ──
      frameCount.value = (frameCount.value + 1) % SEG_SAMPLE_RATE;
      if (frameCount.value !== 0) return;

      try {
        const results = processSegmentationInWorklet(frame, modelRef.current);
        updateSegmentationsWorklet(results);
      } catch (error) {
        logWorklet(`[Worklet] Frame processing error: ${error}`);
        updateSegmentationsWorklet([]);
      }
    },
    [updateSegmentationsWorklet, logWorklet]
  );

  // 카메라 권한 요청
  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);

  // 모델 로딩
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('[Seg] Loading YOLOv8n-seg TFLite model...');
        const model = await loadTensorflowModel(
          require('../assets/yolov8n_seg.tflite'),
          'gpu' as TensorflowModelDelegate
        );
        console.log('[Seg] Model loaded successfully');
        modelRef.current = model;
      } catch (error: any) {
        console.error('[Seg] Model loading error:', error);
        Alert.alert('Model Error', error.message);
      }
    };
    loadModel();

    return () => {
      modelRef.current = null;
      frameCount.value = 0;
      setSegmentations([]);
    };
  }, []);

  return {
    hasPermission,
    segmentations,
    frameProcessor,
    cameraRef,
  };
}

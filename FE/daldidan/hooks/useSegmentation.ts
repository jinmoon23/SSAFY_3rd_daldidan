/**
 * useSegmentation — YOLOv8n-seg 온디바이스 실시간 세그멘테이션 훅
 *
 * 카메라 프레임을 Worklet에서 처리하여 사과의 세그멘테이션 마스크를 실시간 추출.
 * EfficientDet-lite0 기반 useObjectDetection을 대체.
 */

import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import { Camera, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useImageProcessing } from './useImageProcessing';
import { postprocessSegWorklet, SegOutputResult } from './useSegPostprocessing';
import { SegmentationResult } from './types/objectDetection';
import {
  SEG_MODEL_INPUT_SIZE,
  SEG_SAMPLE_RATE,
} from '../constants/segModel';

export function useSegmentation(format: any) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const cameraRef = useRef<Camera>(null);
  const frameCount = Worklets.createSharedValue(0);
  const [segmentations, setSegmentations] = useState<SegmentationResult[]>([]);
  const [hasPermission, setHasPermission] = useState(false);

  const { preprocessFrameForSeg, logWorklet } = useImageProcessing();

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

  // FrameProcessor — 매 N프레임마다 세그멘테이션 추론
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!modelRef.current) return;

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

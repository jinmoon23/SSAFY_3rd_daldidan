// daldidan/components/CameraViewNoDetect.tsx
// Phase 3: 실시간 세그멘테이션 + 터치 당도 예측 버전

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  AppState,
  ActivityIndicator,
} from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useSegmentation } from '../hooks/useSegmentation';
import { useTouchToApple } from '../hooks/useTouchToApple';
import { useSweetnessPredictor } from '../hooks/useSweetnessPredictor';
import { SegmentationResult } from '../hooks/types/objectDetection';
import AppleHint from './AppleHint';
import RealtimeSegOverlay from './RealtimeSegOverlay';
import * as SplashScreen from 'expo-splash-screen';
SplashScreen.preventAutoHideAsync();

export default function CameraView() {
  const device = useCameraDevice('back');
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const [appState, setAppState] = useState('active');

  // 당도가 반영된 세그멘테이션 결과 (모델 출력 + 터치 결과 병합)
  const [enrichedSegs, setEnrichedSegs] = useState<SegmentationResult[]>([]);
  // 현재 분석 중인 사과 id 세트
  const loadingIdsRef = useRef<Set<number>>(new Set());

  // App 상태 변화 감지
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      setAppState(nextAppState);
    });
    return () => subscription.remove();
  }, []);

  // 카메라 설정
  const format =
    device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats[0];
  const fps = format ? Math.min(60, format.maxFps) : 30;

  // Phase 3: 온디바이스 당도 예측
  const sweetness = useSweetnessPredictor();

  // Phase 1: 실시간 세그멘테이션 훅 (+ Phase 3 sweetness config)
  const { hasPermission, segmentations, frameProcessor, cameraRef } =
    useSegmentation(format, {
      sweetnessModelRef: sweetness.sweetnessModelRef,
      cropRequest: sweetness.cropRequest,
      handleFeaturesFromWorklet: sweetness.handleFeaturesFromWorklet,
    });

  // 프레임 크기 (카메라 Landscape 기준)
  const frameSize = {
    width: 1920,
    height: 1080,
  };

  // Phase 2: 터치 → 사과 매핑
  const { findAppleAtTouch } = useTouchToApple({
    segmentations: enrichedSegs,
    screenSize,
    frameSize,
  });

  // 모델 세그멘테이션이 업데이트되면 기존 당도 정보를 보존하면서 병합
  useEffect(() => {
    setEnrichedSegs((prev) => {
      const sweetnessMap = new Map<number, { sweetness?: number; isLoading?: boolean }>();
      for (const p of prev) {
        if (p.sweetness !== undefined || p.isLoading) {
          sweetnessMap.set(p.id, { sweetness: p.sweetness, isLoading: p.isLoading });
        }
      }
      return segmentations.map((seg) => {
        const existing = sweetnessMap.get(seg.id);
        if (existing) {
          return { ...seg, ...existing };
        }
        return seg;
      });
    });
  }, [segmentations]);

  // Phase 3: 당도 예측 결과 수신 → UI 업데이트
  useEffect(() => {
    if (!sweetness.predictionResult) return;
    const { appleId, sweetness: brix } = sweetness.predictionResult;
    setEnrichedSegs((prev) =>
      prev.map((seg) =>
        seg.id === appleId
          ? { ...seg, isLoading: false, sweetness: brix }
          : seg
      )
    );
    loadingIdsRef.current.delete(appleId);
    console.log(`[Phase3] Apple #${appleId}: ${brix.toFixed(2)} Brix`);
  }, [sweetness.predictionResult]);

  // 터치 핸들러: 사과 터치 → 온디바이스 당도 예측 요청
  const handleOverlayTouch = useCallback(
    (screenX: number, screenY: number) => {
      const appleId = findAppleAtTouch(screenX, screenY);
      if (appleId === null) return;

      // 이미 로딩 중이면 무시
      if (loadingIdsRef.current.has(appleId)) return;

      // 터치한 사과의 bbox 가져오기
      const touchedSeg = enrichedSegs.find((s) => s.id === appleId);
      if (!touchedSeg) return;

      console.log(`[Phase3] Apple #${appleId} touched. Requesting on-device prediction...`);

      // 로딩 상태 설정
      loadingIdsRef.current.add(appleId);
      setEnrichedSegs((prev) =>
        prev.map((seg) =>
          seg.id === appleId ? { ...seg, isLoading: true, sweetness: undefined } : seg
        )
      );

      // 프레임 프로세서에 크롭 요청 (다음 프레임에서 처리)
      sweetness.requestPrediction(appleId, touchedSeg.bbox);
    },
    [findAppleAtTouch, enrichedSegs, sweetness]
  );

  const hasApple = enrichedSegs.length > 0;

  useEffect(() => {
    if (device && hasPermission && format) {
      SplashScreen.hideAsync();
    }
  }, [device, hasPermission, format]);

  return (
    <View
      style={StyleSheet.absoluteFill}
      onLayout={(event) => {
        const { width, height } = event.nativeEvent.layout;
        setScreenSize({ width, height });
      }}
    >
      {!hasPermission || !device || !format ? (
        <View style={styles.container}>
          <ActivityIndicator size='large' color='white' />
          <Text style={{ color: 'white', marginTop: 12 }}>
            카메라 설정 또는 권한 확인 중...
          </Text>
        </View>
      ) : (
        <View style={StyleSheet.absoluteFill}>
          {/* Camera — 항상 활성 (자동 촬영 없음) */}
          {appState === 'active' ? (
            <Camera
              ref={cameraRef}
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={true}
              frameProcessor={frameProcessor}
              fps={fps}
              format={format}
              photo={true}
            />
          ) : null}

          {/* 실시간 세그멘테이션 마스크 오버레이 + 터치 인터랙션 */}
          {hasApple &&
          screenSize.width > 0 &&
          screenSize.height > 0 ? (
            <RealtimeSegOverlay
              segmentations={enrichedSegs}
              screenSize={screenSize}
              frameSize={frameSize}
              onTouch={handleOverlayTouch}
            />
          ) : null}

          {/* 사과 미감지 시 힌트 */}
          {!hasApple ? <AppleHint /> : null}
        </View>
      )}
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
  },
});

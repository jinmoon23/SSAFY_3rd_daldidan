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
import { useObjectAnalysis } from '../hooks/useObjectAnalysis';
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

  // Phase 1: 실시간 세그멘테이션 훅
  const { hasPermission, segmentations, frameProcessor, cameraRef } =
    useSegmentation(format);

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

  // Phase 2: 서버 API
  const { sendAnalysisRequest } = useObjectAnalysis();

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

  // 터치 핸들러: 사과 터치 → 프레임 캡처 → 서버 API → 당도 업데이트
  const handleOverlayTouch = useCallback(
    async (screenX: number, screenY: number) => {
      const appleId = findAppleAtTouch(screenX, screenY);
      if (appleId === null) return;

      // 이미 로딩 중이면 무시
      if (loadingIdsRef.current.has(appleId)) return;

      console.log(`[Phase2] Apple #${appleId} touched. Capturing photo...`);

      // 로딩 상태 설정
      loadingIdsRef.current.add(appleId);
      setEnrichedSegs((prev) =>
        prev.map((seg) =>
          seg.id === appleId ? { ...seg, isLoading: true, sweetness: undefined } : seg
        )
      );

      try {
        // 1. 사진 캡처
        if (!cameraRef.current) throw new Error('Camera not available');
        const photo = await cameraRef.current.takePhoto({ flash: 'off' });
        const photoUri = `file://${photo.path}`;
        console.log(`[Phase2] Photo captured: ${photoUri}`);

        // 2. FormData 생성 (전체 이미지 전송 — 서버가 세그멘테이션+당도 예측)
        const formData = new FormData();
        formData.append('image', {
          uri: photoUri,
          name: `apple_${appleId}_${Date.now()}.jpg`,
          type: 'image/jpeg',
        } as any);

        // 3. 서버 API 호출
        console.log(`[Phase2] Sending to server...`);
        const results = await sendAnalysisRequest(formData);
        console.log(`[Phase2] Server response:`, results);

        // 4. 응답에서 당도 추출 (첫 번째 결과 또는 가장 가까운 bbox 매칭)
        let sweetness: number | undefined;
        if (results && results.length > 0) {
          // 터치한 사과의 bbox와 가장 가까운 서버 결과 매칭
          const touchedSeg = enrichedSegs.find((s) => s.id === appleId);
          if (touchedSeg && results.length > 1) {
            // 여러 결과 중 bbox 중심 거리가 가장 가까운 것 선택
            const tcx = (touchedSeg.bbox.xmin + touchedSeg.bbox.xmax) / 2;
            const tcy = (touchedSeg.bbox.ymin + touchedSeg.bbox.ymax) / 2;
            let minDist = Infinity;
            for (const r of results) {
              if (r.sugar_content == null) continue;
              const rcx = (r.bbox.xmin + r.bbox.xmax) / 2;
              const rcy = (r.bbox.ymin + r.bbox.ymax) / 2;
              const dist = Math.sqrt((tcx - rcx) ** 2 + (tcy - rcy) ** 2);
              if (dist < minDist) {
                minDist = dist;
                sweetness = r.sugar_content ?? undefined;
              }
            }
          } else {
            sweetness = results[0].sugar_content ?? undefined;
          }
        }

        // 5. 당도 업데이트
        setEnrichedSegs((prev) =>
          prev.map((seg) =>
            seg.id === appleId ? { ...seg, isLoading: false, sweetness } : seg
          )
        );
        console.log(`[Phase2] Apple #${appleId} sweetness: ${sweetness ?? 'N/A'}`);
      } catch (error: any) {
        console.error(`[Phase2] Analysis failed for apple #${appleId}:`, error.message);
        setEnrichedSegs((prev) =>
          prev.map((seg) =>
            seg.id === appleId ? { ...seg, isLoading: false } : seg
          )
        );
      } finally {
        loadingIdsRef.current.delete(appleId);
      }
    },
    [findAppleAtTouch, cameraRef, sendAnalysisRequest, enrichedSegs]
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

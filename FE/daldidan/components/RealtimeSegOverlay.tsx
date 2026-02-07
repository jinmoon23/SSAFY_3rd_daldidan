/**
 * RealtimeSegOverlay — 실시간 세그멘테이션 마스크 렌더링 + 터치 인터랙션 컴포넌트
 *
 * Skia Canvas로 세그멘테이션 폴리곤을 실시간 렌더링.
 * Phase 2: 터치 → 사과 매핑 → 당도 Tooltip 표시.
 */

import React, { useMemo } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableWithoutFeedback,
  ActivityIndicator,
} from 'react-native';
import {
  Canvas,
  Path,
  Skia,
  SkPath,
  Paint,
  Rect,
  Group,
} from '@shopify/react-native-skia';
import { SegmentationResult } from '../hooks/types/objectDetection';

interface Props {
  segmentations: SegmentationResult[];
  screenSize: { width: number; height: number };
  frameSize: { width: number; height: number };
  onTouch?: (screenX: number, screenY: number) => void;
}

// 사과별 마스크 색상 (반투명)
const MASK_COLORS = [
  'rgba(255, 82, 85, 0.35)',
  'rgba(255, 165, 0, 0.35)',
  'rgba(50, 205, 50, 0.35)',
  'rgba(65, 105, 225, 0.35)',
  'rgba(238, 130, 238, 0.35)',
];

const STROKE_COLORS = [
  'rgba(255, 82, 85, 0.8)',
  'rgba(255, 165, 0, 0.8)',
  'rgba(50, 205, 50, 0.8)',
  'rgba(65, 105, 225, 0.8)',
  'rgba(238, 130, 238, 0.8)',
];

/**
 * 프레임 좌표(Landscape) → 화면 좌표(Portrait) 변환
 *
 * 카메라 프레임은 Landscape (1920×1080)이고 화면은 Portrait이므로
 * 90° 시계방향 회전 + 스케일링이 필요
 */
function transformToScreen(
  frameX: number,
  frameY: number,
  frameW: number,
  frameH: number,
  screenW: number,
  screenH: number
): { x: number; y: number } {
  // 90° 시계방향 회전: (x, y) → (frameH - y, x)
  const rotatedX = frameH - frameY;
  const rotatedY = frameX;

  // 회전 후 이미지 크기: (frameH × frameW)
  const rotatedW = frameH;
  const rotatedH = frameW;

  // 화면에 맞게 스케일링 (높이 기준 fit)
  const scale = screenH / rotatedH;
  const scaledW = rotatedW * scale;
  const offsetX = (screenW - scaledW) / 2;

  return {
    x: rotatedX * scale + offsetX,
    y: rotatedY * scale,
  };
}

function createPolygonPath(
  polygon: number[][],
  frameW: number,
  frameH: number,
  screenW: number,
  screenH: number
): SkPath | null {
  if (polygon.length < 3) return null;

  const path = Skia.Path.Make();
  const first = transformToScreen(
    polygon[0][0],
    polygon[0][1],
    frameW,
    frameH,
    screenW,
    screenH
  );
  path.moveTo(first.x, first.y);

  for (let i = 1; i < polygon.length; i++) {
    const pt = transformToScreen(
      polygon[i][0],
      polygon[i][1],
      frameW,
      frameH,
      screenW,
      screenH
    );
    path.lineTo(pt.x, pt.y);
  }
  path.close();
  return path;
}

export default function RealtimeSegOverlay({
  segmentations,
  screenSize,
  frameSize,
  onTouch,
}: Props) {
  const paths = useMemo(() => {
    if (
      !segmentations ||
      segmentations.length === 0 ||
      screenSize.width === 0 ||
      screenSize.height === 0
    )
      return [];

    return segmentations
      .map((seg, i) => {
        const path = createPolygonPath(
          seg.polygon,
          frameSize.width,
          frameSize.height,
          screenSize.width,
          screenSize.height
        );
        if (!path) return null;
        return {
          path,
          fillColor: MASK_COLORS[i % MASK_COLORS.length],
          strokeColor: STROKE_COLORS[i % STROKE_COLORS.length],
          score: seg.score,
          sweetness: seg.sweetness,
          bbox: seg.bbox,
          id: seg.id,
        };
      })
      .filter(Boolean);
  }, [segmentations, screenSize, frameSize]);

  if (paths.length === 0) return null;

  const handlePress = (event: any) => {
    if (!onTouch) return;
    const { locationX, locationY } = event.nativeEvent;
    onTouch(locationX, locationY);
  };

  return (
    <TouchableWithoutFeedback onPress={handlePress}>
      <View style={StyleSheet.absoluteFill}>
        {/* Skia Canvas — pointerEvents="none"로 터치가 부모 View를 통과 */}
        <Canvas style={StyleSheet.absoluteFill} pointerEvents="none">
          {paths.map((item, i) => {
            if (!item) return null;
            return (
              <Group key={`seg-${i}`}>
                {/* 마스크 영역 (반투명 채우기) */}
                <Path
                  path={item.path}
                  color={item.fillColor}
                  style="fill"
                />
                {/* 윤곽선 */}
                <Path
                  path={item.path}
                  color={item.strokeColor}
                  style="stroke"
                  strokeWidth={2}
                />
              </Group>
            );
          })}
        </Canvas>

        {/* 당도 Tooltip 또는 로딩 스피너 (터치 후 표시) */}
        {segmentations.map((seg, i) => {
          const showLoading = seg.isLoading === true;
          const showSweetness =
            seg.sweetness !== undefined && seg.sweetness !== null;
          if (!showLoading && !showSweetness) return null;

          const center = transformToScreen(
            (seg.bbox.xmin + seg.bbox.xmax) / 2,
            (seg.bbox.ymin + seg.bbox.ymax) / 2,
            frameSize.width,
            frameSize.height,
            screenSize.width,
            screenSize.height
          );

          return (
            <View
              key={`tooltip-${i}`}
              style={[
                styles.tooltip,
                { left: center.x - 40, top: center.y - 20 },
              ]}
              pointerEvents="none"
            >
              {showLoading ? (
                <View style={styles.loadingRow}>
                  <ActivityIndicator size="small" color="white" />
                  <Text style={styles.loadingText}>분석 중...</Text>
                </View>
              ) : (
                <Text style={styles.sweetnessText}>
                  🍎 {seg.sweetness!.toFixed(1)} Brix
                </Text>
              )}
            </View>
          );
        })}
      </View>
    </TouchableWithoutFeedback>
  );
}

const styles = StyleSheet.create({
  tooltip: {
    position: 'absolute',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 14,
    zIndex: 10,
    minWidth: 80,
    alignItems: 'center',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  loadingText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  sweetnessText: {
    color: 'white',
    fontSize: 15,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});

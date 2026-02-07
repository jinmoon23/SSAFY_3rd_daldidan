# 📱 온디바이스 실시간 사과 객체 인식 시스템

> **EfficientDet-lite0 (TFLite) + React Native Vision Camera FrameProcessor 기반**
> 서버 통신 없이 모바일 디바이스에서 실시간으로 사과를 탐지하는 온디바이스 AI 파이프라인

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처 전체 구조](#2-아키텍처-전체-구조)
3. [핵심 파일 구조 및 역할](#3-핵심-파일-구조-및-역할)
4. [모델 상세 (EfficientDet-lite0)](#4-모델-상세-efficientdet-lite0)
5. [FrameProcessor 파이프라인](#5-frameprocessor-파이프라인)
6. [이미지 전처리 (useImageProcessing)](#6-이미지-전처리-useimageprocessing)
7. [추론 및 후처리 (useObjectDetection)](#7-추론-및-후처리-useobjectdetection)
8. [좌표 변환 시스템](#8-좌표-변환-시스템)
9. [자동 촬영 시퀀스 (Auto-Capture FSM)](#9-자동-촬영-시퀀스-auto-capture-fsm)
10. [UI 컴포넌트 연동](#10-ui-컴포넌트-연동)
11. [스레드 모델 및 성능 최적화](#11-스레드-모델-및-성능-최적화)
12. [빌드 설정 및 네이티브 연동](#12-빌드-설정-및-네이티브-연동)
13. [주요 상수 및 설정값 참조표](#13-주요-상수-및-설정값-참조표)
14. [데이터 흐름 시퀀스 다이어그램](#14-데이터-흐름-시퀀스-다이어그램)
15. [트러블슈팅 및 설계 결정](#15-트러블슈팅-및-설계-결정)

---

## 1. 시스템 개요

본 시스템은 **서버 요청 없이 모바일 디바이스 자체에서 카메라 프레임을 실시간 분석**하여 사과 객체를 탐지합니다. 탐지된 사과가 일정 시간 유지되면 자동으로 고해상도 사진을 촬영하고, 해당 사진을 서버 AI 파이프라인에 전송하여 당도를 예측하는 **하이브리드 AI 아키텍처**의 1단계(온디바이스) 부분입니다.

### 왜 온디바이스인가?

| 관점 | 서버사이드 탐지 | 온디바이스 탐지 (본 프로젝트) |
|------|----------------|----------------------------|
| **지연시간** | 네트워크 왕복 200~500ms+ | 프레임 당 ~30ms (GPU delegate) |
| **네트워크 의존** | 필수 (오프라인 불가) | 불필요 (완전 오프라인 탐지) |
| **배터리** | 통신 비용 높음 | 모델 경량화로 최적화 |
| **사용자 경험** | 프레임 끊김 | 부드러운 실시간 피드백 |
| **역할** | - | 사과 존재 여부 판단 → 자동 촬영 트리거 |

> 온디바이스 모델은 **정밀 당도 예측이 아닌, 사과의 존재 여부를 빠르게 판단**하는 역할에 집중합니다.
> 당도 예측은 서버사이드의 YOLOv8l-seg + CNN+MLP Fusion 모델이 담당합니다.

---

## 2. 아키텍처 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    온디바이스 객체 인식 아키텍처                           │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────┐
                        │  Camera Hardware     │
                        │  (Back Camera)       │
                        │  1920×1080 @ 60fps   │
                        └──────────┬──────────┘
                                   │
                                   │ Raw Frame (YUV/RGB)
                                   │
                        ┌──────────▼──────────┐
                        │  Vision Camera       │
                        │  react-native-       │
                        │  vision-camera ^4.6  │
                        └──────────┬──────────┘
                                   │
                                   │ useFrameProcessor
                                   │ (Worklet Thread — UI 비차단)
                                   │
              ┌────────────────────▼────────────────────┐
              │          FrameProcessor Pipeline          │
              │  ┌─────────────────────────────────────┐ │
              │  │ ① Frame Sampling (15프레임당 1회)    │ │
              │  │    SAMPLE_RATE = 15                  │ │
              │  │    → 60fps 기준 약 4fps 추론          │ │
              │  └──────────────┬──────────────────────┘ │
              │                 │                         │
              │  ┌──────────────▼──────────────────────┐ │
              │  │ ② 전처리 (useImageProcessing)        │ │
              │  │    vision-camera-resize-plugin       │ │
              │  │                                      │ │
              │  │    Raw Frame (1920×1080)              │ │
              │  │         │                             │ │
              │  │         ▼ 중앙 정사각형 크롭           │ │
              │  │    1080×1080 (short side)             │ │
              │  │         │                             │ │
              │  │         ▼ 리사이즈                     │ │
              │  │    320×320 RGB uint8                  │ │
              │  └──────────────┬──────────────────────┘ │
              │                 │                         │
              │  ┌──────────────▼──────────────────────┐ │
              │  │ ③ TFLite 추론                        │ │
              │  │    react-native-fast-tflite          │ │
              │  │    model.runSync([input])             │ │
              │  │    GPU Delegate 가속                  │ │
              │  │                                      │ │
              │  │    Output[0]: boxes (Float32)         │ │
              │  │    Output[1]: classes (Float32)       │ │
              │  │    Output[2]: scores (Float32)        │ │
              │  │    Output[3]: numDetections (Float32) │ │
              │  └──────────────┬──────────────────────┘ │
              │                 │                         │
              │  ┌──────────────▼──────────────────────┐ │
              │  │ ④ 후처리                             │ │
              │  │    a. class_id=52 (사과) 필터링       │ │
              │  │    b. score ≥ 0.3 (CONFIDENCE)       │ │
              │  │    c. NMS (IoU threshold = 0.4)      │ │
              │  │    d. 좌표 유효성 검증 (0~1 범위)     │ │
              │  │    e. 정규화좌표 → 절대좌표 변환       │ │
              │  └──────────────┬──────────────────────┘ │
              │                 │                         │
              └─────────────────┼─────────────────────────┘
                                │
                   Worklets.createRunOnJS()
                   (Worklet Thread → JS Thread 전환)
                                │
              ┌─────────────────▼─────────────────────┐
              │         JS Thread (React State)         │
              │                                         │
              │  setDetections(Detection[])              │
              │       │                                  │
              │       ├── hasApple 상태 업데이트          │
              │       ├── 자동 촬영 시퀀스 트리거         │
              │       └── UI 오버레이 업데이트            │
              └─────────────────────────────────────────┘
```

---

## 3. 핵심 파일 구조 및 역할

```
FE/daldidan/
├── hooks/
│   ├── useObjectDetection.ts      ⭐ 핵심: TFLite 로딩 + FrameProcessor + NMS + 후처리
│   ├── useImageProcessing.ts      ⭐ 핵심: Worklet 내 프레임 전처리 (크롭+리사이즈)
│   ├── useShake.ts                   가속도계 기반 흔들기 감지 → 리셋 트리거
│   ├── useAnalysisApiHandler.ts      서버 API 상태 관리 (분석 트리거/결과/에러)
│   ├── useObjectAnalysis.ts          fetch POST /predict (순수 API 통신)
│   └── types/
│       └── objectDetection.ts     ⭐ 타입 정의: Detection, AnalyzedObjectResult, etc.
│
├── components/
│   ├── CameraViewNoDetect.tsx     ⭐ 메인 카메라 뷰 + 상태 머신 + 자동 촬영 시퀀스
│   ├── DetectionOverlay.tsx          실시간 탐지 결과 Glow 오버레이 (현재 미사용)
│   ├── AppleHint.tsx                 사과 미감지 시 "사과를 비춰주세요" 안내
│   ├── CaptureOverlay.tsx            촬영 카운트다운 중 캐릭터+플래시 애니메이션
│   ├── AppleProcessing.tsx           API 분석 중 "사과즙 짜는 중..." 로딩
│   └── AnalyzedResultOverlay.tsx     분석 완료 후 세그멘테이션 + 당도 시각화
│
├── constants/
│   ├── model.ts                   ⭐ MODEL_INPUT_SIZE(320), CONFIDENCE_THRESHOLD(0.3)
│   └── api.ts                        API 엔드포인트 정의
│
├── assets/
│   ├── model.tflite               ⭐ EfficientDet-lite0 양자화 모델 파일
│   ├── 1.tflite                      대체/실험 모델 파일
│   ├── 2.tflite                      대체/실험 모델 파일
│   ├── lottie/                       애니메이션 JSON (scan, flash, juice, tap)
│   ├── sounds/countdown.mp3          촬영 카운트다운 사운드
│   └── images/                       캐릭터 이미지, 아이콘 등
│
├── metro.config.js                   .tflite 확장자 번들링 설정
├── babel.config.js                   worklets-core + reanimated 플러그인 설정
└── app.json                          네이티브 플러그인 설정 (GPU 라이브러리 등)
```

---

## 4. 모델 상세 (EfficientDet-lite0)

### 4-1. 모델 선택 이유

| 후보 모델 | 크기 | 추론 속도 | 정확도 | 선택 |
|-----------|------|-----------|--------|------|
| YOLOv8n (TFLite) | ~6.2MB | ~50ms | 높음 | ❌ 출력 형식 후처리 복잡 |
| EfficientDet-lite0 | ~4.4MB | ~30ms | 충분 | ✅ **최종 채택** |
| MobileNet SSD v2 | ~6.9MB | ~40ms | 보통 | ❌ |

> 초기에는 YOLOv8n을 TFLite로 변환하여 시도했으나 (팀원 3명 참여), 최종적으로 **EfficientDet-lite0**이
> React Native 환경에서의 호환성, GPU delegate 지원, 출력 텐서 구조의 단순함 면에서 가장 적합하다고 판단하여 채택.

### 4-2. 모델 스펙

```
모델명:          EfficientDet-lite0 (양자화, TFLite)
파일:            assets/model.tflite
입력 텐서:       [1, 320, 320, 3]  — RGB uint8
출력 텐서 4개:
  ├── outputs[0]: boxes         — Float32Array [N×4] (y1, x1, y2, x2, 정규화 0~1)
  ├── outputs[1]: classes       — Float32Array [N]   (COCO 80 클래스 ID)
  ├── outputs[2]: scores        — Float32Array [N]   (신뢰도 0.0~1.0)
  └── outputs[3]: numDetections — Float32Array [1]   (유효 탐지 수)
학습 데이터:     COCO 2017 (80 클래스, 사과 = class_id 52)
양자화:          INT8 또는 Float16 (경량화)
가속:            GPU Delegate (OpenCL — Pixel, Mali GPU)
```

### 4-3. COCO 클래스 ID 매핑

```
class_id = 52  →  "apple" (사과)   ✅ 본 프로젝트 타겟
class_id = 59  →  "donut" (도넛)   — 보조 감지 (둥근 형태 오인 대비)
```

### 4-4. 모델 로딩 코드 (useObjectDetection.ts)

```typescript
// hooks/useObjectDetection.ts:272-295
useEffect(() => {
  const loadModel = async () => {
    try {
      const model = await loadTensorflowModel(
        require('../assets/model.tflite'),   // 번들된 TFLite 파일
        'gpu' as TensorflowModelDelegate     // GPU Delegate 가속
      );
      modelRef.current = model;
    } catch (error: any) {
      Alert.alert('Model Error', error.message);
    }
  };
  loadModel();

  return () => {
    modelRef.current = null;  // 메모리 해제
    frameCount.value = 0;
    setDetections([]);
    setDetectionResults([]);
  };
}, []);
```

**핵심 포인트:**
- `loadTensorflowModel`은 `react-native-fast-tflite` 라이브러리의 함수
- `'gpu'` delegate를 명시하여 **OpenCL 기반 GPU 가속**을 활성화
- `modelRef`를 `useRef`로 관리하여 **Worklet 스레드에서 안전하게 접근**
- 컴포넌트 언마운트 시 `modelRef.current = null`로 명시적 정리

### 4-5. GPU Delegate 네이티브 설정 (app.json)

```json
// app.json:59-64
[
  "react-native-fast-tflite",
  {
    "enableAndroidGpuLibraries": [
      "libOpenCL-pixel.so",   // Google Pixel 시리즈
      "libGLES_mali.so"       // Samsung Galaxy (Mali GPU)
    ]
  }
]
```

**지원 GPU:**
- **Google Pixel**: Qualcomm Adreno → `libOpenCL-pixel.so`
- **Samsung Galaxy**: ARM Mali → `libGLES_mali.so`
- GPU 미지원 디바이스에서는 자동으로 CPU fallback

---

## 5. FrameProcessor 파이프라인

### 5-1. FrameProcessor란?

React Native Vision Camera의 **FrameProcessor**는 카메라 프레임을 **Worklet 스레드**에서 동기적으로 처리할 수 있는 메커니즘입니다. UI 스레드나 JS 스레드를 차단하지 않으면서, C++/GPU 수준의 성능으로 프레임 단위 연산을 수행합니다.

### 5-2. 프레임 샘플링 전략

```typescript
// hooks/useObjectDetection.ts:228-263
const SAMPLE_RATE = 15;

const frameProcessor = useFrameProcessor(
  async (frame) => {
    'worklet';
    if (!modelRef.current) return;

    // 15프레임마다 1번만 추론 실행
    frameCount.value = (frameCount.value + 1) % SAMPLE_RATE;
    if (frameCount.value !== 0) return;

    const detections = processDetectionsInWorklet(frame, modelRef.current);

    if (detections && detections.length > 0) {
      const items = detections.map((detection) => ({
        detection: { ...detection },
        timestamp: Date.now(),
      }));
      runOnJSThread(items);          // JS 스레드로 결과 전달
      updateDetectionsWorklet(detections);  // React State 업데이트
    } else {
      updateDetectionsWorklet([]);
    }
  },
  [updateDetectionsWorklet, runOnJSThread, logWorklet]
);
```

**샘플링 계산:**

```
카메라 FPS:     최대 60fps (format에 따라 가변)
SAMPLE_RATE:    15
실제 추론 FPS:  60 / 15 = 약 4fps
추론 1회 소요:  약 25~35ms (GPU delegate 기준)

→ 나머지 14프레임은 카메라 프리뷰만 표시 (매끄러운 UX)
→ 4fps면 약 250ms 간격으로 사과 위치 업데이트 (충분한 반응성)
```

### 5-3. 카메라 설정

```typescript
// components/CameraViewNoDetect.tsx:117-123
const format = device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats[0];
const fps = format ? Math.min(60, format.maxFps) : 30;

<Camera
  ref={cameraRef}
  style={StyleSheet.absoluteFill}
  device={device}
  isActive={!isAnalyzing && analyzedResults === null}  // API 분석 중에는 카메라 일시정지
  frameProcessor={frameProcessor}
  fps={fps}
  format={format}
  photo={true}  // takePhoto() 기능 활성화
/>
```

**설정 상세:**
- `useCameraDevice('back')`: 후면 카메라 사용
- **60fps 우선 선택**: `formats.find(f => f.maxFps >= 60)`
- `isActive` prop으로 **카메라 생명주기 자동 관리** (분석 중 → 카메라 정지 → 배터리 절약)
- `photo={true}`: takePhoto() 고해상도 캡처를 위한 모드 활성화

---

## 6. 이미지 전처리 (useImageProcessing)

### 6-1. 전처리 파이프라인 상세

```typescript
// hooks/useImageProcessing.ts:17-35
const preprocessFrame = (frame: any, targetSize: number) => {
  'worklet';
  const shortSide = Math.min(frame.width, frame.height);  // 1080 (portrait에서)
  const cropX = (frame.width - shortSide) / 2;             // 중앙 크롭 오프셋 X
  const cropY = (frame.height - shortSide) / 2;             // 중앙 크롭 오프셋 Y

  return resize(frame, {
    scale: { width: targetSize, height: targetSize },  // 320×320
    pixelFormat: 'rgb',     // RGB 채널
    dataType: 'uint8',      // 0~255 정수
    crop: {
      x: cropX,             // 중앙에서 시작
      y: cropY,
      width: shortSide,     // 정사각형 크롭
      height: shortSide,
    },
  });
};
```

### 6-2. 전처리 시각화

```
 카메라 원본 프레임 (1920 × 1080, Landscape 기준)
 ┌─────────────────────────────────────────────────────┐
 │         │                            │               │
 │  420px  │     중앙 크롭 영역          │    420px      │
 │ 버림    │     1080 × 1080            │    버림       │
 │         │                            │               │
 │◄───────►│◄──────────────────────────►│◄─────────────►│
 │ cropX   │        shortSide           │               │
 │ = 420   │        = 1080              │               │
 └─────────────────────────────────────────────────────┘

                         │
                         ▼  resize (vision-camera-resize-plugin)
                         │  네이티브 C++ 레벨에서 고속 처리

                  ┌──────────────┐
                  │              │
                  │  320 × 320   │
                  │  RGB uint8   │
                  │              │
                  └──────────────┘
                         │
                         ▼  model.runSync([input])
```

**핵심 설계 결정:**
- **정사각형 크롭**: 모델 입력이 정사각형(320×320)이므로, 원본 프레임에서 먼저 정사각형으로 크롭하여 **가로세로 비율 왜곡 방지**
- **중앙 크롭**: 사과는 보통 화면 중앙에 위치한다는 가정 (UX 가이드와 일치)
- **네이티브 resize**: `vision-camera-resize-plugin`은 C++ 레벨에서 resize를 수행하여 JS 오버헤드 최소화

### 6-3. 감지 영역 크롭 (개별 객체 추출)

```typescript
// hooks/useImageProcessing.ts:37-92
const extractCroppedData = async (frame, detection) => {
  'worklet';
  const { x, y, width, height } = detection;

  // 경계 보정
  const safeX = Math.max(0, Math.min(x, frame.width - 1));
  const safeY = Math.max(0, Math.min(y, frame.height - 1));
  const safeWidth = Math.min(width, frame.width - safeX);
  const safeHeight = Math.min(height, frame.height - safeY);

  // 최소 크기 제한 (20×20px 미만은 무시)
  if (safeWidth < 20 || safeHeight < 20) return null;

  // 최대 100×100px으로 제한 (메모리 절약)
  const maxSize = 100;
  const resized = resize(frame, {
    scale: {
      width: Math.min(safeWidth, maxSize),
      height: Math.min(safeHeight, maxSize),
    },
    pixelFormat: 'rgb',
    dataType: 'uint8',
    crop: { x: safeX, y: safeY, width: safeWidth, height: safeHeight },
  });

  return {
    data: Array.from(new Uint8Array(resized)),
    width: Math.min(safeWidth, maxSize),
    height: Math.min(safeHeight, maxSize),
    isJPEG: false,
  };
};
```

---

## 7. 추론 및 후처리 (useObjectDetection)

### 7-1. 추론 + 후처리 전체 흐름

```typescript
// hooks/useObjectDetection.ts:76-150
const processDetectionsInWorklet = (frame, model) => {
  'worklet';
  // ① 전처리
  const resized = preprocessFrame(frame, MODEL_INPUT_SIZE);  // 320×320

  // ② TFLite 동기 추론
  const outputs = model.runSync([resized]);
  const boxes = outputs[0] as Float32Array;         // [N×4]: y1, x1, y2, x2
  const classes = outputs[1] as Float32Array;        // [N]: class_id
  const scores = outputs[2] as Float32Array;         // [N]: confidence
  const numDetections = outputs[3] as Float32Array;  // [1]: 유효 탐지 수

  // ③ 사과 필터링 (class_id=52, score≥0.3)
  for (let i = 0; i < totalDetections; i++) {
    if (Math.round(classes[i]) !== 52) continue;
    if (scores[i] < CONFIDENCE_THRESHOLD) continue;
    // → filteredBoxes, filteredScores에 추가
  }

  // ④ Non-Max Suppression (IoU=0.4)
  const keepIdx = nonMaxSuppression(filteredBoxes, filteredScores, 0.4);

  // ⑤ 좌표 변환: 정규화(0~1) → 절대 픽셀 좌표
  for (const idx of keepIdx) {
    const cropSize = 1080;
    const cropOffsetX = (1920 - cropSize) / 2;  // = 420

    const x = clamp(x1 * cropSize + cropOffsetX, 0, frame.width);
    const y = clamp(y1 * cropSize, 0, frame.height);
    const width = clamp((x2 - x1) * cropSize, 0, frame.width - x);
    const height = clamp((y2 - y1) * cropSize, 0, frame.height - y);

    detected.push({ x, y, width, height, score, class_id: 52 });
  }

  return detected;
};
```

### 7-2. Non-Max Suppression (NMS) 상세

NMS는 **같은 사과에 대한 중복 바운딩 박스를 제거**하는 알고리즘입니다.

```
 NMS 전 (중복 박스 존재)              NMS 후 (최고 점수 1개만 유지)

 ┌──────────────┐                    ┌──────────────┐
 │  ┌─────────┐ │  score=0.92       │              │
 │  │ ┌──────┐│ │  score=0.87       │   🍎         │  score=0.92
 │  │ │  🍎  ││ │  score=0.71       │  최종 박스    │
 │  │ └──────┘│ │                    │              │
 │  └─────────┘ │                    └──────────────┘
 └──────────────┘
```

```typescript
// hooks/useObjectDetection.ts:39-74
function nonMaxSuppression(boxes, scores, iouThreshold = 1) {
  'worklet';
  // 점수 내림차순 정렬
  const idxs = boxes.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];

  for (const i of idxs) {
    let shouldKeep = true;
    for (const j of keep) {
      // IoU (Intersection over Union) 계산
      const xx1 = Math.max(boxes[i].x1, boxes[j].x1);
      const yy1 = Math.max(boxes[i].y1, boxes[j].y1);
      const xx2 = Math.min(boxes[i].x2, boxes[j].x2);
      const yy2 = Math.min(boxes[i].y2, boxes[j].y2);
      const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
      const areaI = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
      const areaJ = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
      const iou = inter / (areaI + areaJ - inter);

      if (iou > iouThreshold) { shouldKeep = false; break; }
    }
    if (shouldKeep) keep.push(i);
  }
  return keep;
}
```

**IoU = 0.4 의미:**
- 두 박스가 40% 이상 겹치면 **같은 객체**로 간주
- 점수가 더 낮은 박스를 제거
- 값이 너무 낮으면(0.2) 가까이 있는 별개의 사과도 제거됨
- 값이 너무 높으면(0.8) 같은 사과에 여러 박스가 남음

### 7-3. 중복 탐지 방지 (Grid-based Deduplication)

같은 위치의 사과가 매 프레임 반복 처리되는 것을 방지합니다.

```typescript
// hooks/useObjectDetection.ts:152-198
function getGridKey(detection) {
  const grid = 80;  // 80px 그리드 단위
  return [
    detection.class_id,
    Math.round((detection.x + detection.width / 2) / grid) * grid,   // 중심X 양자화
    Math.round((detection.y + detection.height / 2) / grid) * grid,  // 중심Y 양자화
  ].join('_');
}

// "52_480_320" 같은 키로 중복 체크
// DUPLICATE_TIMEOUT = 5000ms (5초) 이내 동일 위치 요청 무시
```

```
 ┌────────────────────────────────────────┐
 │  80px 그리드 양자화                      │
 │                                        │
 │   ┌──┬──┬──┬──┬──┬──┐                  │
 │   │  │  │  │  │  │  │                  │
 │   ├──┼──┼──┼──┼──┼──┤                  │
 │   │  │  │🍎│  │  │  │ ← 그리드 셀에    │
 │   ├──┼──┼──┼──┼──┼──┤   사과 중심 매핑  │
 │   │  │  │  │  │  │  │                  │
 │   ├──┼──┼──┼──┼──┼──┤                  │
 │   │  │  │  │  │  │  │                  │
 │   └──┴──┴──┴──┴──┴──┘                  │
 │                                        │
 │  같은 그리드 셀 → 같은 키 → 5초간 무시  │
 └────────────────────────────────────────┘
```

---

## 8. 좌표 변환 시스템

### 8-1. 좌표 변환이 필요한 이유

```
모델 출력 좌표:  정규화 (0.0 ~ 1.0)  →  320×320 기준
카메라 프레임:   1920×1080 (Landscape)
전처리 크롭:    중앙 1080×1080
화면 표시:      Portrait (예: 360×780)

→ 4단계 좌표계를 거쳐야 정확한 화면 위치를 얻을 수 있음
```

### 8-2. 좌표 변환 수학

```
                 모델 출력 좌표 (정규화 0~1)
                       │
                       │ × cropSize (1080)
                       ▼
                 크롭 영역 좌표 (0~1080)
                       │
                       │ + cropOffsetX (420)
                       ▼
                 원본 프레임 좌표 (0~1920, 0~1080)
                       │
                       │ 90° 회전 (Landscape → Portrait)
                       │ + 스케일링 (frame → screen)
                       ▼
                 화면 좌표 (screenWidth × screenHeight)
```

**코드에서의 구현:**

```typescript
// hooks/useObjectDetection.ts:126-133
const cropSize = 1080;                        // 전처리에서 크롭한 정사각형 크기
const cropOffsetX = (1920 - cropSize) / 2;    // = 420px (좌우 대칭 오프셋)

// 정규화 → 크롭 영역 → 원본 프레임 좌표
const x = clamp(x1 * cropSize + cropOffsetX, 0, frame.width);
const y = clamp(y1 * cropSize, 0, frame.height);
const width = clamp((x2 - x1) * cropSize, 0, frame.width - x);
const height = clamp((y2 - y1) * cropSize, 0, frame.height - y);
```

### 8-3. 화면 표시용 좌표 변환 (DetectionOverlay)

실시간 오버레이 표시를 위해 카메라 프레임 좌표를 화면 좌표로 변환합니다.

```typescript
// components/DetectionOverlay.tsx:58-77
const frameW = 1920;
const frameH = 1080;

// ① Landscape→Portrait 90° 회전
const rotated = {
  x: detection.y,
  y: frameW - detection.x - detection.width,
  width: detection.height,
  height: detection.width,
};

// ② 프레임→화면 스케일링
const scaleX = screenW / frameH;   // 360 / 1080
const scaleY = screenH / frameW;   // 780 / 1920

const x = rotated.x * scaleX;
const y = rotated.y * scaleY;
const width = rotated.width * scaleX;
const height = rotated.height * scaleY;
```

```
  카메라 프레임 (Landscape)          화면 (Portrait)
  ┌──────────────────────┐          ┌────────────┐
  │   1920 × 1080        │          │ 360 × 780  │
  │                      │   90°    │            │
  │    ┌──┐              │  회전    │   ┌──┐     │
  │    │🍎│              │ ──────→  │   │🍎│     │
  │    └──┘              │  + 스케일 │   └──┘     │
  │                      │          │            │
  └──────────────────────┘          └────────────┘
```

---

## 9. 자동 촬영 시퀀스 (Auto-Capture FSM)

온디바이스 탐지의 궁극적 목적은 **사과가 감지되면 자동으로 고해상도 사진을 촬영**하여 서버 API로 전송하는 것입니다. 이 과정은 유한 상태 머신(FSM) 패턴으로 구현되어 있습니다.

### 9-1. 상태 전이 다이어그램

```
                    사과 감지
 ┌──────────┐  ─────────────────→  ┌─────────────────┐
 │           │                     │                  │
 │   IDLE    │                     │  CAPTURE_OVERLAY │
 │  (탐지중) │  ←────────────────  │  (카운트다운 +   │
 │           │   사과 사라짐 or    │   사운드 재생)   │
 └──────────┘    촬영 실패         └────────┬────────┘
      ▲                                     │
      │                              사운드 완료 +
      │                              사과 여전히 감지
      │                                     │
      │                                     ▼
      │                            ┌─────────────────┐
      │          흔들기(Shake)      │                  │
      │  ◄─────────────────────── │   ANALYZING      │
      │                            │  (API 호출 +     │
      │                            │   사과즙 로딩)   │
      │                            └────────┬────────┘
      │                                     │
      │                              API 응답 수신
      │                                     │
      │                                     ▼
      │                            ┌─────────────────┐
      │          흔들기(Shake)      │                  │
      └──────────────────────────  │   RESULT         │
                                   │  (당도 결과 표시) │
                                   │                  │
                                   └─────────────────┘
```

### 9-2. 자동 촬영 트리거 조건

```typescript
// components/CameraViewNoDetect.tsx:144-163
useEffect(() => {
  // ❌ 사과 사라지면 카운트다운 즉시 중단
  if (!hasApple && countdown !== null) {
    setCountdown(null);
    clearInterval(countdownTimer.current);
  }

  // ❌ 아래 조건 중 하나라도 해당하면 촬영 시작 안 함
  if (
    !hasApple ||                    // 사과 미감지
    isAnalyzing ||                  // API 분석 진행 중
    analyzedResults !== null ||     // 이전 결과 아직 표시 중
    !autoCaptureEnabled ||          // 자동 캡처 비활성
    justReset.current               // 방금 리셋됨 (2초 쿨다운)
  ) return;

  // ✅ 모든 조건 충족 → 촬영 시퀀스 시작
  startCaptureSequence();
}, [detections, hasApple, isAnalyzing, analyzedResults, autoCaptureEnabled]);
```

### 9-3. 촬영 시퀀스 상세

```typescript
// components/CameraViewNoDetect.tsx:235-284
const startCaptureSequence = () => {
  // Guard: 이미 촬영 중이거나 조건 불충족 시 무시
  if (isAnalyzing || analyzedResults !== null || showCaptureImage ||
      capturingRef.current || freezeDetection || !hasApple) return;

  capturingRef.current = true;
  setShowCaptureImage(true);      // CaptureOverlay 표시 (캐릭터 애니메이션)
  setFreezeDetection(true);       // 탐지 일시 동결

  // 100ms 간격으로 사과 감지 상태 모니터링
  const appleDetectionCheck = setInterval(() => {
    if (!hasAppleRef.current) {
      // 사과가 사라지면 모든 동작 중단 (사운드 포함)
      clearInterval(appleDetectionCheck);
      setShowCaptureImage(false);
      capturingRef.current = false;
      countdownSoundRef.current?.stop();
    }
  }, 100);

  // 카운트다운 사운드 재생 → 완료 시 촬영
  countdownSoundRef.current?.play((success) => {
    clearInterval(appleDetectionCheck);
    if (success && hasAppleRef.current) {
      // 사운드 재생 완료 + 사과 여전히 감지 → 촬영 실행
      handleCaptureAndAnalyze().then(() => {
        setShowCaptureImage(false);
        capturingRef.current = false;
      });
    } else {
      // 실패 (사과 사라짐 or 사운드 에러) → 시퀀스 취소
      setShowCaptureImage(false);
      capturingRef.current = false;
    }
  });
};
```

### 9-4. 고해상도 사진 촬영

```typescript
// components/CameraViewNoDetect.tsx:166-227
const handleCaptureAndAnalyze = useCallback(async () => {
  if (!cameraRef.current || isAnalyzing) return;

  const photo = await cameraRef.current.takePhoto({
    qualityPrioritization: 'speed',      // 속도 우선 (당도 예측에는 충분한 품질)
    enableShutterAnimation: false,       // 셔터 애니메이션 없음 (자동 촬영이므로)
  });

  const uri = `file://${photo.path}`;
  const photoOriginalWidth = photo.width;   // 원본 해상도 보존
  const photoOriginalHeight = photo.height;

  // 서버 API 전송 트리거 (원본 해상도 메타데이터 함께 전달)
  await triggerAnalysis(uri, photoOriginalWidth, photoOriginalHeight);
}, [isAnalyzing, triggerAnalysis, cameraRef]);
```

### 9-5. 흔들기(Shake)로 리셋

```typescript
// hooks/useShake.ts
export function useShake(onShake, threshold = 1.5, interval = 1000) {
  useEffect(() => {
    let lastShakeTime = 0;
    const subscription = Accelerometer.addListener(({ x, y, z }) => {
      const acceleration = Math.sqrt(x * x + y * y + z * z);
      if (acceleration > threshold && now - lastShakeTime > interval) {
        lastShakeTime = now;
        onShake();
      }
    });
    Accelerometer.setUpdateInterval(100);  // 100ms 간격 샘플링
    return () => subscription.remove();
  }, [onShake, threshold, interval]);
}

// components/CameraViewNoDetect.tsx:294-310
useShake(() => {
  if (analysisFinished) {
    justReset.current = true;    // 2초간 자동 캡처 방지
    resetAnalysis();             // 결과 초기화 → 카메라 재개
    setTimeout(() => {
      justReset.current = false; // 2초 후 자동 캡처 재허용
    }, 2000);
  }
}, 2.0, 700);  // threshold=2.0, debounce=700ms
```

---

## 10. UI 컴포넌트 연동

### 10-1. 상태별 UI 렌더링 매핑

```
┌─────────────────────────────────────────────────────────────────┐
│                    상태별 UI 표시 매트릭스                        │
├──────────────────┬─────────────────────────────────────────────┤
│ 상태              │ 표시 컴포넌트                                │
├──────────────────┼─────────────────────────────────────────────┤
│ 사과 미감지       │ AppleHint ("🍎 사과를 비춰주세요")           │
│                  │ + Lottie 스캔 애니메이션                     │
├──────────────────┼─────────────────────────────────────────────┤
│ 사과 감지 +       │ CaptureOverlay (캐릭터 애니메이션            │
│ 카운트다운 중     │ + Lottie 플래시 + 사운드 재생)               │
├──────────────────┼─────────────────────────────────────────────┤
│ API 분석 중       │ AppleProcessing ("사과즙 짜는 중...")         │
│                  │ + Lottie 사과즙 애니메이션                   │
├──────────────────┼─────────────────────────────────────────────┤
│ 결과 표시 중      │ AnalyzedResultOverlay                       │
│                  │ (Skia 세그멘테이션 + 당도 + 왕관 + 슬라이더) │
├──────────────────┼─────────────────────────────────────────────┤
│ 탐지 0개 결과     │ "객체 인식 결과 없음" 메시지                  │
└──────────────────┴─────────────────────────────────────────────┘
```

### 10-2. 렌더링 조건 로직

```typescript
// components/CameraViewNoDetect.tsx:333-386
<View style={StyleSheet.absoluteFill}>
  {/* ① 카메라 프리뷰 (분석 중에는 정지) */}
  <Camera isActive={!isAnalyzing && analyzedResults === null} ... />

  {/* ② 결과 오버레이 (분석 완료 + 결과 있음 + 화면 크기 유효) */}
  {analysisFinished && analyzedResults?.length > 0 &&
   originalImageSize && screenSize.width > 0 ? (
    <AnalyzedResultOverlay ... />
  ) : null}

  {/* ③ 결과 없음 메시지 */}
  {analysisFinished && analyzedResults?.length === 0 ? (
    <Text>"객체 인식 결과 없음"</Text>
  ) : null}

  {/* ④ 분석 중 로딩 */}
  {isAnalyzing && <AppleProcessing status='juicing' />}

  {/* ⑤ 사과 미감지 힌트 (다른 상태가 모두 아닐 때만) */}
  {detections.length === 0 && !isAnalyzing &&
   analyzedResults === null && !showCaptureImage && !freezeDetection ? (
    <AppleHint />
  ) : null}

  {/* ⑥ 촬영 카운트다운 오버레이 */}
  <CaptureOverlay visible={showCaptureImage && !isAnalyzing} ... />
</View>
```

---

## 11. 스레드 모델 및 성능 최적화

### 11-1. 3-스레드 아키텍처

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    React Native 스레드 모델                    │
 └─────────────────────────────────────────────────────────────┘

 ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
 │  UI Thread       │   │  JS Thread       │   │  Worklet Thread  │
 │  ──────────────  │   │  ──────────────  │   │  ────────────── │
 │  • 화면 렌더링   │   │  • React 상태    │   │  • 프레임 수신  │
 │  • 터치 이벤트   │   │  • useEffect     │   │  • 전처리       │
 │  • 네이티브 UI   │   │  • API 호출      │   │  • TFLite 추론  │
 │  • 애니메이션    │   │  • 상태 업데이트  │   │  • NMS 후처리   │
 │                  │   │                  │   │  • 좌표 변환    │
 │  ❌ 추론 없음    │   │  ❌ 추론 없음    │   │  ✅ 추론 전담   │
 │  → 항상 60fps   │   │  → 블로킹 없음   │   │  → 병렬 처리   │
 └─────────────────┘   └────────┬─────────┘   └────────┬────────┘
                                │                       │
                                │  Worklets.createRunOnJS()
                                │◄──────────────────────┘
                                │  (탐지 결과 배열 전달)
```

### 11-2. Worklet ↔ JS 스레드 통신

```typescript
// Worklet → JS (탐지 결과 전달)
const updateDetectionsWorklet = useRef(
  Worklets.createRunOnJS((data: Detection[]) => {
    setDetections(data);  // React State 업데이트는 JS Thread에서
  })
).current;

// 중요: useRef로 래핑하여 콜백이 매 렌더마다 재생성되지 않도록 함
// → 메모리 누수 및 클로저 문제 방지
```

### 11-3. 성능 최적화 기법 요약

| 기법 | 구현 위치 | 효과 |
|------|----------|------|
| **GPU Delegate** | `loadTensorflowModel(..., 'gpu')` | 추론 속도 2~5배 향상 |
| **프레임 샘플링** | `SAMPLE_RATE = 15` (매 15프레임당 1회) | 불필요한 추론 93% 절감 |
| **네이티브 리사이즈** | `vision-camera-resize-plugin` | JS 오버헤드 제거 |
| **Worklet 스레드 분리** | `'worklet'` directive | UI 60fps 유지 |
| **useRef 콜백 안정화** | `useRef(Worklets.createRunOnJS(...)).current` | GC 압력 감소 |
| **SharedValue** | `Worklets.createSharedValue(0)` | 스레드 간 저비용 값 공유 |
| **Grid 중복 제거** | `getGridKey()` + 5초 타임아웃 | 불필요한 재처리 방지 |
| **카메라 isActive 제어** | `isActive={!isAnalyzing && ...}` | 분석 중 카메라 정지 → 배터리 절약 |
| **class_id 사전 필터링** | NMS 전에 class_id=52만 필터 | NMS 연산량 감소 |
| **clamp 함수** | Worklet 내 인라인 처리 | 좌표 경계 오류 방지 |

---

## 12. 빌드 설정 및 네이티브 연동

### 12-1. Metro Bundler 설정

```javascript
// metro.config.js
const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('tflite');  // .tflite 파일을 에셋으로 번들링
module.exports = config;
```

> `metro.config.js`에 `.tflite` 확장자를 등록하지 않으면, `require('../assets/model.tflite')` 시 번들링 에러가 발생합니다.

### 12-2. Babel 플러그인

```javascript
// babel.config.js
plugins: [
  ['react-native-worklets-core/plugin', { processNestedWorklets: true }],
  ['react-native-reanimated/plugin', { relativeSourceLocation: true }],
]
```

- **worklets-core/plugin**: `'worklet'` 디렉티브를 가진 함수를 Worklet 스레드용으로 변환
- **processNestedWorklets: true**: 중첩된 Worklet 함수 지원 (preprocessFrame 내부 호출 등)
- **reanimated/plugin**: `useAnimatedStyle`, `useSharedValue` 등을 네이티브 코드로 변환

### 12-3. Expo 네이티브 플러그인 설정

```json
// app.json
{
  "plugins": [
    ["expo-build-properties", {
      "android": { "minSdkVersion": 26 }   // Android 8.0+ (TFLite GPU 지원)
    }],
    ["react-native-vision-camera", {
      "cameraPermission": "Allow $(PRODUCT_NAME) to access your camera"
    }],
    ["react-native-fast-tflite", {
      "enableAndroidGpuLibraries": [
        "libOpenCL-pixel.so",    // Pixel GPU
        "libGLES_mali.so"        // Mali GPU
      ]
    }]
  ],
  "newArchEnabled": true    // React Native New Architecture 활성화
}
```

### 12-4. 주요 의존성

```json
{
  "react-native-vision-camera": "^4.6.4",        // 카메라 + FrameProcessor
  "react-native-fast-tflite": "^1.6.1",          // TFLite 모델 로딩/추론
  "react-native-worklets-core": "^1.5.0",        // Worklet 스레드 관리
  "vision-camera-resize-plugin": "^3.2.0",       // 네이티브 프레임 리사이즈
  "react-native-reanimated": "~3.16.1",          // 네이티브 애니메이션
  "expo-sensors": "~14.0.2",                     // 가속도계 (Shake 감지)
  "react-native-sound": "^0.11.2",               // 카운트다운 사운드
  "lottie-react-native": "7.1.0",                // Lottie 애니메이션
  "@shopify/react-native-skia": "1.5.0"          // Skia Canvas 렌더링
}
```

---

## 13. 주요 상수 및 설정값 참조표

| 상수/설정 | 값 | 파일 | 설명 |
|-----------|-----|------|------|
| `MODEL_INPUT_SIZE` | `320` | `constants/model.ts` | 모델 입력 크기 (320×320) |
| `CONFIDENCE_THRESHOLD` | `0.3` | `constants/model.ts` | 최소 신뢰도 (30%) |
| `SAMPLE_RATE` | `15` | `useObjectDetection.ts` | N프레임당 1회 추론 |
| NMS IoU Threshold | `0.4` | `useObjectDetection.ts` | NMS 중복 제거 기준 |
| Apple class_id | `52` | `useObjectDetection.ts` | COCO 사과 클래스 ID |
| Grid 양자화 크기 | `80px` | `useObjectDetection.ts` | 중복 탐지 방지 그리드 |
| `DUPLICATE_TIMEOUT` | `5000ms` | `useObjectDetection.ts` | 동일 위치 재처리 쿨다운 |
| Shake threshold | `2.0` | `CameraViewNoDetect.tsx` | 흔들기 감지 가속도 임계값 |
| Shake debounce | `700ms` | `CameraViewNoDetect.tsx` | 흔들기 연속 방지 간격 |
| justReset cooldown | `2000ms` | `CameraViewNoDetect.tsx` | 리셋 후 자동 캡처 재허용 |
| Smooth factor | `0.2` | `DetectionOverlay.tsx` | 박스 위치 보간 계수 |
| Min crop size | `20×20px` | `useImageProcessing.ts` | 최소 크롭 크기 (미만 무시) |
| Max crop size | `100×100px` | `useImageProcessing.ts` | 최대 크롭 리사이즈 크기 |
| Camera FPS | `≤60` | `CameraViewNoDetect.tsx` | 카메라 최대 프레임 레이트 |
| GPU Libraries | `OpenCL-pixel`, `GLES_mali` | `app.json` | GPU delegate 네이티브 라이브러리 |
| Min Android SDK | `26` | `app.json` | Android 8.0+ 필수 |

---

## 14. 데이터 흐름 시퀀스 다이어그램

```
 시간→  ════════════════════════════════════════════════════════════════════════►

 Camera  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
 60fps   ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓

 Sample  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ★  ·  ·  ·  ·  ·  ·  ·
 (15당1) ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ↓  ·  ·  ·  ·  ·  ·  ·

 Worklet                                              ┌──────┐
 Thread                                               │전처리│
                                                      │ 추론 │
                                                      │후처리│
                                                      └──┬───┘
                                                         │ createRunOnJS
 JS                                                      ▼
 Thread                                           setDetections([🍎])
                                                         │
                                                         ▼
                                                  hasApple = true
                                                         │
                                                         ▼
                                                  startCaptureSequence()
                                                         │
                                                  ┌──────▼──────┐
                                                  │ CaptureOverlay│
                                                  │ 사운드 재생   │
                                                  │ (카운트다운)   │
                                                  └──────┬──────┘
                                                         │ 사운드 완료
                                                         ▼
                                                  takePhoto()
                                                  { path, width, height }
                                                         │
                                                         ▼
                                                  triggerAnalysis(uri, w, h)
                                                         │
                                                  POST /predict ────→ [서버]
                                                         │
                                                  isAnalyzing = true
                                                  Camera.isActive = false
                                                         │
                                                  ◄────── JSON Response
                                                         │
                                                  analyzedResults = [...]
                                                  isAnalyzing = false
                                                         │
                                                         ▼
                                                  AnalyzedResultOverlay
                                                  (세그멘테이션 + 당도)
                                                         │
                                                  ◄────── 흔들기 (Shake)
                                                         │
                                                  resetAnalysis()
                                                  Camera.isActive = true
                                                         │
                                                         ▼
                                                  다시 IDLE 상태로...
```

---

## 15. 트러블슈팅 및 설계 결정

### 15-1. 왜 YOLOv8n에서 EfficientDet-lite0으로 변경했나?

| 문제 | 원인 | 해결 |
|------|------|------|
| YOLOv8n TFLite 변환 후 출력 형식 비호환 | YOLO의 출력 텐서가 표준 SSD 형식과 달라 react-native-fast-tflite에서 파싱 불가 | EfficientDet-lite0 채택 (표준 SSD 출력 형식) |
| YOLOv8n GPU delegate 미지원 이슈 | 일부 연산이 GPU delegate에서 미지원 | EfficientDet-lite0은 모든 연산이 GPU delegate 호환 |

### 15-2. Worklet 콜백 안정성

| 문제 | 원인 | 해결 |
|------|------|------|
| `Worklets.createRunOnJS` 매 렌더마다 재생성 | 콜백이 클로저를 캡처하여 stale state 참조 | `useRef().current`로 콜백 한 번만 생성 |
| hasApple 상태 stale closure 문제 | `setInterval` 내부에서 hasApple이 최초값으로 고정 | `hasAppleRef.current`로 최신값 참조 |

### 15-3. 좌표계 불일치

| 문제 | 원인 | 해결 |
|------|------|------|
| 탐지 박스가 화면 위치와 불일치 | 카메라 Landscape → 앱 Portrait 회전 미반영 | 90° 좌표 회전 변환 적용 |
| 크롭 영역 밖 좌표 | 정규화 좌표가 크롭 중심이 아닌 원본 기준 | `cropOffsetX` 오프셋 보정 |
| 바운딩 박스가 화면 밖으로 나감 | 좌표 경계 미검증 | `clamp()` 함수로 경계 제한 |

### 15-4. 자동 촬영 경쟁 조건 (Race Condition)

| 문제 | 원인 | 해결 |
|------|------|------|
| 촬영 시퀀스 중복 실행 | detections 변경 시 useEffect 재실행 | `capturingRef.current` 뮤텍스 |
| 리셋 직후 즉시 재촬영 | 사과 여전히 감지 중이므로 | `justReset.current` + 2초 쿨다운 |
| 카운트다운 중 사과 사라짐 | 사운드 재생이 비동기 | 100ms 인터벌로 `hasAppleRef.current` 모니터링 |

---

> **본 문서는 달디단 프로젝트의 온디바이스 실시간 사과 객체 인식 시스템을 포트폴리오 관점에서 상세히 기술한 README입니다.**
> **EfficientDet-lite0 TFLite 모델을 React Native Vision Camera의 FrameProcessor와 결합하여,
> 서버 통신 없이 모바일 디바이스에서 4fps 실시간 사과 탐지를 구현한 하이브리드 AI 아키텍처의 핵심 부분입니다.**

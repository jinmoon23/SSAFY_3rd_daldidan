# 🍎 AI 기반 사과 당도 예측 서비스, **달디단**

## 📅 프로젝트 진행 기간

**2025.04.14 ~ 2025.05.22 (6주)**

## 📑 프로젝트 기획 및 회의 노션

[🔗 달디단 프로젝트 Notion 바로가기](https://laced-brand-a7e.notion.site/3-1d5cb8da549480b79cbce87e4c00a1c3)

---

## 🌟 서비스 한줄 소개

> **완전 온디바이스 AI 기반 사과 당도 예측 서비스**
> 서버 없이, 네트워크 없이 — 스마트폰 카메라만으로 사과의 당도를 실시간 예측

---

## 📝 개요

**달디단**(**Daldidan**)은 스마트폰 카메라를 사과에 비추면 **실시간으로 세그멘테이션 마스크**가 표시되고, 사과를 **터치하면 즉시 당도(Brix)를 예측**하는 완전 온디바이스 AI 서비스입니다.

YOLOv8n-seg로 실시간 사과 감지 및 세그멘테이션을 수행하고, EfficientNet-B0 + MLP Fusion 모델로 당도를 예측합니다. **모든 AI 추론이 디바이스에서 실행**되므로 네트워크 연결 없이 완전한 오프라인 환경에서도 동작합니다.

---

## 🎯 프로젝트 목표

- 📷 카메라 화면에서 사과를 **실시간 세그멘테이션** (~10 FPS)
- � 터치한 사과의 당도를 **온디바이스에서 즉시 예측** (<500ms)
- 🔄 **멀티프레임 앙상블** (5프레임 중앙값)로 예측 안정성 확보
- � **Fingerprint 재인식**으로 카메라 이동 후에도 당도 자동 복원
- � **완전 오프라인** 동작 (서버/네트워크 불필요)

---

## 🛠️ 기술 스택

### Frontend (Mobile)

| 기술 | 용도 |
|---|---|
| React Native 0.76 + Expo 52 | 크로스 플랫폼 모바일 앱 |
| react-native-vision-camera 4.6 | 카메라 접근 + Frame Processor |
| react-native-fast-tflite 1.6 | TFLite 모델 온디바이스 추론 (GPU delegate) |
| react-native-worklets-core 1.5 | 프레임 프로세서 Worklet 실행 |
| vision-camera-resize-plugin 3.2 | Worklet 내 이미지 리사이즈/크롭 |
| @shopify/react-native-skia 1.5 | 세그멘테이션 마스크 및 bbox 실시간 렌더링 |
| expo-router 4.0 | 파일 기반 라우팅 |

### AI / ML

| 기술 | 용도 |
|---|---|
| YOLOv8n-seg (TFLite, 온디바이스) | 실시간 사과 감지 + 세그멘테이션 |
| EfficientNet-B0 (TFLite, 온디바이스) | CNN 특징 1280차원 추출 |
| MLP 1286→128→1 (JS 구현) | 당도(Brix) 회귀 예측 |
| YOLOv8l-seg (PyTorch, 서버) | 서버사이드 사과 감지 (Legacy) |
| PyTorch · timm · scikit-learn | 모델 학습 및 실험 |

### Backend (Legacy)

| 기술 | 용도 |
|---|---|
| FastAPI (Python) | REST API 서버 |
| Docker | 백엔드 컨테이너화 |

### Infra

| 기술 | 용도 |
|---|---|
| AWS EC2 | 서버 배포 환경 |
| Jenkins | GitLab 연동 CI/CD 파이프라인 |
| Nginx | 리버스 프록시 + HTTPS |
| GitLab | 소스 코드 관리 |
| Expo EAS | 모바일 앱 빌드/배포 |

---

## 🚀 주요 기능

### 1. 실시간 세그멘테이션 마스크

- YOLOv8n-seg TFLite 모델로 **~10 FPS** 실시간 추론
- Skia Canvas로 사과 영역에 **폴리곤 마스크 + 바운딩 박스** 실시간 렌더링
- Stable ID 추적 (IoU 매칭)으로 프레임 간 사과 ID 유지

### 2. 터치 기반 온디바이스 당도 예측

- 사과를 터치하면 **EfficientNet-B0 + Manual Features + MLP** 파이프라인으로 즉시 당도 예측
- CNN 임베딩 1280차원 + 수작업 특징 6차원 = **1286차원 Fusion Vector**
- MLP(1286→128→1) 회귀로 Brix 값 예측

### 3. 멀티프레임 앙상블

- 5프레임에 걸쳐 독립 예측 후 **중앙값(median)** 으로 최종 당도 확정
- 프레임 간 노이즈를 제거하여 예측 안정성 향상

### 4. Fingerprint 재인식

- 당도가 확정된 사과의 CNN 특징벡터를 **Fingerprint 캐시**에 저장
- 카메라를 이동했다 다시 비추면 **코사인 유사도 + 공간 거리** 기반 자동 매칭
- 재터치 없이 이전 당도 결과를 즉시 복원

### 5. 멀티 사과 동시 처리

- 여러 사과를 개별적으로 터치하여 **독립 앙상블 버퍼**에서 동시 예측
- 사과별 cropQueue 큐잉으로 프레임 프로세서에서 순차 처리

### 6. 완전 오프라인 동작

- 네트워크 연결 없이 모든 AI 추론이 디바이스에서 실행
- 로그인 불필요, 앱 실행 즉시 사용 가능

---

## 🧪 핵심 기술 상세

### 1. 온디바이스 추론 파이프라인 (3-모델 분리 전략)

```
Camera Frame (1920×1080, 30fps)
    │
    ▼ FrameProcessor (Worklet Thread)
    │
    ├─ Model 1: YOLOv8n-seg (매 3프레임마다 추론)
    │   전처리: 640×640 float32 (0~1)
    │   출력: bbox + 160×160 마스크 계수
    │   후처리: sigmoid → NMS → mask×proto → 폴리곤
    │
    ├─ Model 2: EfficientNet-B0 (터치 시 크롭 큐에서 소비)
    │   입력: 224×224 float32 (ImageNet 정규화)
    │   출력: 1280차원 CNN 특징벡터
    │
    └─ Manual Features (64×64 uint8 크롭)
        RGB, YCbCr, GLCM 기반 6차원 수작업 특징
    │
    ▼ JS Thread
    │
    ├─ StandardScaler 정규화
    ├─ MLP Head: 1286 → 128 (ReLU) → 1
    ├─ 멀티프레임 앙상블 (5회 중앙값)
    └─ Fingerprint 캐시/매칭
```

### 2. YOLOv8n-seg 실시간 세그멘테이션

| 항목 | 세부 내용 |
|---|---|
| **모델** | YOLOv8n-seg (TFLite float32, 13.8MB) |
| **입력** | [1, 640, 640, 3] float32 |
| **출력** | Output[0]: [1, 116, 8400] (bbox+class+mask coeffs), Output[1]: [1, 160, 160, 32] (proto masks) |
| **후처리** | sigmoid → class=47(apple) 필터링 → NMS(IoU=0.45) → mask×proto → threshold → 폴리곤 추출 |
| **추론 빈도** | 매 3프레임 (카메라 30fps 기준 ~10fps) |
| **가속** | GPU Delegate (OpenCL) |

### 3. CNN + MLP Fusion 당도 예측

| 항목 | 세부 내용 |
|---|---|
| **CNN** | EfficientNet-B0 (TFLite float16, 8MB) → 1280차원 |
| **수작업 특징** | Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow(GLCM) |
| **정규화** | StandardScaler (mean/std 하드코딩) |
| **MLP** | FC1(1286→128, ReLU) → FC2(128→1) → Brix |
| **앙상블** | 5프레임 독립 예측 후 중앙값 |
| **MLP 가중치** | mlpWeights.bin (3.7MB, JSON 비동기 로드) |

### 4. 프론트엔드 시각화

| 항목 | 세부 내용 |
|---|---|
| **마스크 렌더링** | Skia Canvas + Path 폴리곤 + RoundedRect bbox |
| **좌표 변환** | 프레임(Landscape) → 90° 회전 → 화면(Portrait) 스케일링 |
| **터치 판정** | Ray Casting point-in-polygon + bbox fallback |
| **당도 표시** | 로딩 스피너 → Brix 값 Tooltip (사과 중심 위치) |

---

## 🌐 시스템 아키텍처

### 현재 운영 아키텍처 (Phase 3.5 — 완전 온디바이스)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Mobile App (React Native + Expo)                       │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Worklet Thread (Frame Processor)               │   │
│   │                                                                   │   │
│   │   Camera Frame ──→ YOLOv8n-seg ──→ NMS + Mask ──→ Polygons      │   │
│   │        │                                                          │   │
│   │        ├──→ EfficientNet-B0 (224×224 crop) ──→ CNN [1280]        │   │
│   │        └──→ Manual Features (64×64 crop)  ──→ Features [6]       │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │ Worklets.createRunOnJS()               │
│   ┌────────────────────────────▼────────────────────────────────────┐   │
│   │                    JS Thread                                      │   │
│   │                                                                   │   │
│   │   Stable ID (IoU) ──→ MLP (1286→128→1) ──→ Ensemble (median)    │   │
│   │                              │                                    │   │
│   │                     Fingerprint Cache ←─→ Re-ID 매칭              │   │
│   └────────────────────────────┬────────────────────────────────────┘   │
│                                │                                        │
│   ┌────────────────────────────▼────────────────────────────────────┐   │
│   │                    UI Thread                                      │   │
│   │                                                                   │   │
│   │   Skia Canvas (마스크/bbox) ──→ 당도 Tooltip ──→ 터치 인터랙션   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   📦 온디바이스 모델: yolov8n_seg.tflite (13.8MB)                       │
│                       efficientnet_b0_apple.tflite (8MB)                │
│                       mlpWeights.bin (3.7MB)                            │
└─────────────────────────────────────────────────────────────────────────┘

        ※ 서버 연결 불필요 — 완전 오프라인 동작
```

### Legacy 아키텍처 (Phase 1 — 서버 의존)

![시스템 아키텍쳐](./readme_assets/system_architecture.jpg)

---

## 📈 진화 단계

| Phase | 방식 | 상태 |
|---|---|---|
| **Phase 1** | 서버 전송 방식: 캡처 → REST API → YOLOv8l + CNN+MLP (서버) → 결과 수신 | Legacy |
| **Phase 2** | 온디바이스 감지 + 서버 예측: EfficientDet (온디바이스) → 크롭 → API | Legacy |
| **Phase 3** | 완전 온디바이스: YOLOv8n-seg + EfficientNet-B0 + MLP 전부 디바이스 실행 | 구현 완료 |
| **Phase 3.5** | 멀티프레임 앙상블 + Fingerprint 재인식 + 멀티사과 동시 처리 | **현재 운영** |

---

## 📊 성능 목표

| 지표 | 목표 | 비고 |
|---|---|---|
| 세그멘테이션 추론 | <80ms/프레임 | GPU Delegate |
| 세그멘테이션 FPS | ~10fps | SAMPLE_RATE=3 |
| 당도 예측 총시간 | <500ms | 터치→결과 |
| 모델 총 크기 | <40MB | 3개 모델 합산 (25.5MB) |
| UI 프레임레이트 | ≥30fps | 카메라 프리뷰 |
| 당도 MAE | ≤0.8 Brix | 서버 대비 약간 하락 허용 |

---

## 📸 서비스 화면 예시

- 실시간 세그멘테이션 마스크 화면
- 터치 시 당도 예측 로딩
- 당도 결과 Tooltip 표시

---

## 👨‍👩‍👧‍👦 팀원

| 이름   | 역할                        |
| ------ | --------------------------- |
| 최진문(팀장) | Frontend, AI, Infra |
| 박수민 | Frontend, AI               |
| 이원재 | Frontend, AI               |
| 하건수 | Frontend, AI               |
| 전민경 | Backend, AI, Infra        |
| 정한균 | Backend, AI                |

## 📌 역할 및 담당 업무

### 🎨 Front-End

#### 최진문(팀장) [Frontend, AI, Infra]

- **온디바이스 AI 파이프라인 전체 설계 및 구현 (Phase 1 → 3.5)**
- YOLOv8n-seg TFLite 온디바이스 실시간 세그멘테이션 구현
- EfficientNet-B0 + MLP 온디바이스 당도 예측 파이프라인 구현
- 멀티프레임 앙상블 (5프레임 중앙값) 설계 및 구현
- Fingerprint 재인식 시스템 (CNN 코사인 유사도 + 공간 거리 매칭)
- 멀티 사과 동시 처리 아키텍처 (독립 cropQueue / ensembleMap)
- Manual Features JS/Worklet 포팅 (RGB, YCbCr, GLCM)
- Stable ID 추적 (IoU 매칭) 구현
- 실시간 Skia Canvas 마스크/bbox 렌더링
- 터치→사과 매핑 (Ray Casting point-in-polygon)
- 모델 변환 스크립트 작성 (PyTorch → ONNX → TFLite)
- MLP 가중치 / Scaler 값 추출 스크립트
- Infra (Expo EAS 앱 배포)

#### 박수민 [Frontend]

- 프론트측 사과 객체 인식 모델
- 초기 당도 예측 모델 개발(XGBoost)
- 초기 프론트측 사과 객체 인식 모델(YOLOv8n)
- UX/UI 디자인
- 당도 페이지 구현

#### 이원재 [Frontend]

- 프론트측 사과 객체 인식 모델
- 초기 프론트측 사과 객체 인식 모델(YOLOv8n)
- 로딩 페이지 구현
- 당도 페이지 구현

#### 하건수 [Frontend]

- 프론트측 사과 객체 인식 모델
- 초기 프론트측 사과 객체 인식 모델(YOLOv8n)
- UX/UI 디자인
- WebSocket 구현
- 당도 페이지 구현

### 🖥️ Back-End

#### 전민경 [Backend]

- 당도 예측 모델 개발 (CNN + 수작업 특징 기반 Fusion 모델 구현, Fine-tuning 및 Feature 확장 실험)
- 초기 당도 예측 모델 개발(Linear Regression)
- API 구현
- Infra (백엔드 배포)

#### 정한균 [Backend]

- 백엔드측 사과 객체 인식 모델
- 초기 당도 예측 모델 개발(EfficientNet-B0+LightGBM, LightGBM)
- FastAPI 구조 설계
- API 구현

---

## 📌 향후 개선 방향

- 사과 품종 확대 (후지 → 홍로, 아리수, 시나노골드 등 다양한 품종 지원)
- 사과 신선도 예측 기능 추가 (부패 감지, 저장일 추정 등)
- 기타 과일(수박, 귤 등)의 당도 예측 기능 추가
- 당도 기록/저장 기능 및 통계 대시보드
- 육류 이미지 기반 고기 부위 판별 및 등급 분류 기능 확장

---

## 📂 프로젝트 구조

### Frontend

```
📁 FE/daldidan/
├── 📁 app/                              # Expo Router
│   ├── _layout.tsx                      #   ThemeProvider + Stack
│   └── index.tsx                        #   → CameraViewNoDetect
│
├── 📁 components/                       # UI 컴포넌트
│   ├── CameraViewNoDetect.tsx           #   메인 오케스트레이터
│   ├── RealtimeSegOverlay.tsx           #   Skia 마스크/bbox/터치/당도 Tooltip
│   ├── AppleHint.tsx                    #   사과 미감지 힌트
│   ├── AnalyzedResultOverlay.tsx        #   API 결과 오버레이 (Legacy)
│   └── ...                              #   Toast, Button 등 UI
│
├── 📁 hooks/                            # 커스텀 훅
│   ├── useSweetnessPredictor.ts         #   ★ 당도 예측 파이프라인
│   │                                    #     EfficientNet + MLP
│   │                                    #     멀티프레임 앙상블
│   │                                    #     Fingerprint 캐시/매칭
│   ├── useSegmentation.ts               #   ★ YOLOv8n-seg 실시간 추론
│   │                                    #     Frame Processor + Stable ID
│   ├── useSegPostprocessing.ts          #   YOLOv8 후처리 (NMS, 마스크)
│   ├── useManualFeatures.ts             #   수동 특징 6개 (RGB/YCbCr/GLCM)
│   ├── useImageProcessing.ts            #   프레임 리사이즈/크롭 유틸
│   ├── useTouchToApple.ts               #   터치→사과 매핑 (Ray Casting)
│   ├── useObjectDetection.ts            #   EfficientDet (Legacy)
│   └── types/objectDetection.ts         #   타입 정의
│
├── 📁 constants/                        # 상수 정의
│   ├── segModel.ts                      #   YOLOv8n-seg 파라미터
│   ├── scalerValues.json                #   StandardScaler mean/std
│   └── model.ts                         #   EfficientDet 파라미터 (Legacy)
│
├── 📁 assets/                           # 모델 & 리소스
│   ├── yolov8n_seg.tflite               #   세그멘테이션 모델 (13.8MB)
│   ├── efficientnet_b0_apple.tflite     #   CNN 피처 추출 (8MB)
│   ├── mlpWeights.bin                   #   MLP 가중치 (3.7MB)
│   └── ...                              #   폰트, 로티, 사운드
│
├── 📁 android/                          # 안드로이드 네이티브 설정
├── app.json
├── package.json
└── tsconfig.json
```

### Backend

```
📁 BE/
├── 📁 be/                               # Gateway 서버
│   ├── main.py                          #   FastAPI 앱
│   ├── api/v1/routes.py                 #   /health, /dummy_predict
│   ├── schemas/                         #   API/Socket 스키마
│   └── Dockerfile                       #   컨테이너화
│
└── 📁 ai/                               # AI 추론 서버
    ├── main.py                          #   FastAPI 앱
    ├── api/v1/routes.py                 #   /health, /predict
    └── services/
        ├── predict_service.py           #   당도 추론 디스패처
        ├── detect_service.py            #   사과 감지 디스패처
        ├── cnn_feature_maskcrop_seg/    #   ★ CNN+MLP Fusion 모델
        │   ├── train.py                 #     학습 스크립트
        │   ├── fusion_model.py          #     모델 정의
        │   ├── predictor.py             #     추론기
        │   └── extract_features.py      #     특징 추출
        └── yolov8/                      #   YOLO 감지 엔진
            ├── inference/               #     TFLite/PyTorch 백엔드
            └── models/                  #     모델 가중치
```

### 모델 변환 스크립트

```
📁 scripts/
├── convert_efficientnet_to_tflite.py    #   .pth → ONNX → onnx2tf → TFLite
├── convert_yolov8n_seg.py              #   YOLOv8 → TFLite
└── extract_mlp_weights.py              #   PyTorch MLP → JSON 가중치 추출
```

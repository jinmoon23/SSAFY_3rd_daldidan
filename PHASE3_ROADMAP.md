# 🍎 Phase 3: 완전 온디바이스 실시간 세그멘테이션 + 당도 예측 로드맵

> **최종 목표**: 서버 의존 완전 제거. 카메라로 사과를 비추면 즉시 실시간 마스크 표시,
> 사용자가 마스크된 사과를 터치하면 온디바이스에서 당도를 즉시 예측

---

## 1. 현재 vs 목표

```
현재: 사과 감지(bbox) → 2~3초 대기 → 자동촬영 → 서버 전송 → 세그멘테이션+당도 수신
목표: 사과 감지(mask) → 즉시 실시간 마스크 → 터치 → 온디바이스 당도 예측 → 즉시 표시
```

| 항목 | 현재 | 목표 |
|------|------|------|
| 인식 | Bounding Box (EfficientDet) | Segmentation Mask (YOLOv8n-seg) |
| 촬영 | 2~3초 후 자동 | 없음 (실시간 마스크) |
| 당도 | 서버 API (2~5초) | 온디바이스 (<500ms) |
| 네트워크 | 필수 | 불필요 (완전 오프라인) |
| UX | 수동 없음 (자동) | 터치하여 개별 사과 조회 |

---

## 2. 실현 가능성: ✅ 가능

### 이미 보유 중인 자산

| 자산 | 상태 | 활용 |
|------|------|------|
| `yolov8n_seg_float32.tflite` | ✅ 이미 존재 | 온디바이스 세그멘테이션 |
| `yolov8n_seg_float16.tflite` | ✅ 이미 존재 | 경량 버전 |
| `postprocess_seg()` Python | ✅ 로직 완성 | JS Worklet 포팅 |
| FusionModel `.pth` | ✅ 학습 완료 | TFLite 변환 |
| `scaler.pkl` | ✅ 존재 | mean/std → JS 상수 |
| `extract_features()` | ✅ 완성 | JS 포팅 |

### 서버 FusionModel 구조 (분석 완료)

```
FusionModel (cnn_feature_maskcrop_seg)
├── EfficientNet-B0 (timm) → 1280차원 특징벡터
├── Manual Features (6차원)
│   ├── Rn: R/(R+G+B)          ← JS 구현 쉬움
│   ├── C: 1-R/255             ← JS 구현 쉬움
│   ├── ycbcr_diff: Cb-Cr      ← JS 구현 쉬움
│   ├── ycbcr_norm: Cb/(Cb+Cr) ← JS 구현 쉬움
│   ├── cat02_first: 0.0       ← 미사용 (그대로)
│   └── cluster_shadow: GLCM   ← 간소화 가능
├── StandardScaler 정규화
├── MLP: Linear(1286,128) → ReLU → Linear(128,1)
└── 푸른 사과 보정: Lab 색공간 (a_mean < 133, b_mean > 130)
```

### 핵심 전략: 모델 분리

```
Model 1 (실시간, Worklet): YOLOv8n-seg TFLite → bbox + mask
Model 2 (터치 시, JS):     EfficientNet-B0 TFLite → 1280 features
MLP Head:                   JS 직접 구현 (행렬곱 2회)
Manual Features:            JS 직접 구현 (터치 시 1회만 실행)
Scaler:                     mean/std 하드코딩
```

---

## 3. 전체 아키텍처

```
Camera (60fps)
    │
    ▼ FrameProcessor (Worklet Thread)
    │
    ├─ 전처리: 1920×1080 → 640×640
    ├─ YOLOv8n-seg TFLite 추론
    ├─ 후처리: sigmoid → NMS → mask_coeff×proto → 폴리곤
    │
    ▼ createRunOnJS()
    │
JS Thread
    ├─ setSegmentations([{bbox, polygon, score}, ...])
    ├─ Skia Canvas 실시간 마스크 렌더링
    │
    ├─── 사용자 터치 ───┐
    │                    ▼
    │            point-in-polygon 판정
    │                    │
    │            사과 영역 크롭 (224×224)
    │                    │
    │            ┌───────┴────────┐
    │            │ 푸른 사과 체크   │ → Lab 색공간
    │            └───────┬────────┘
    │                    │ (아니면)
    │            Manual Features (JS)
    │            EfficientNet-B0 TFLite → 1280 features
    │            MLP (JS 행렬곱) → 당도 (Brix)
    │                    │
    │            당도 Tooltip 표시
    └────────────────────┘
```

---

## 4. 3단계 구현 로드맵

### Phase 1: 실시간 세그멘테이션 마스크 (2~3주)

**목표**: 카메라에 사과를 비추면 즉시 세그멘테이션 마스크가 실시간 표시

#### Step 1-1: YOLOv8n-seg TFLite 모델 검증

```bash
# 기존 TFLite 모델의 입출력 텐서 확인
python3 -c "
import tensorflow as tf
interp = tf.lite.Interpreter('BE/ai/services/yolov8/models/yolov8n_seg_float32.tflite')
interp.allocate_tensors()
for t in interp.get_input_details(): print('IN:', t['shape'], t['dtype'])
for t in interp.get_output_details(): print('OUT:', t['shape'], t['dtype'])
"
# 예상: IN: [1,640,640,3], OUT[0]: [1,116,8400], OUT[1]: [1,160,160,32]
```

> ⚠️ 텐서 shape 확인이 최우선. 형태에 따라 후처리 코드가 달라짐.

#### Step 1-2: 전처리 변경

- `MODEL_INPUT_SIZE` 320 → 640 (YOLOv8 입력)
- `pixelFormat: 'rgb'`, `dataType: 'float32'` (0~1 정규화)

#### Step 1-3: 후처리 Worklet 포팅

서버 `postprocess_seg()` 로직을 JS Worklet으로 포팅:
1. sigmoid + score 필터링 (class=47, conf>0.25)
2. NMS (IoU=0.45)
3. `mask_coefficients[1×32] × proto[32×25600]` → 160×160 마스크
4. threshold(0.5) → 바이너리 → 폴리곤 좌표 추출

#### Step 1-4: Skia 실시간 마스크 렌더링

- `RealtimeSegmentationOverlay.tsx` 신규 컴포넌트
- 세그멘테이션 폴리곤 → Skia Path → 반투명 컬러 오버레이
- 좌표 변환: 640×640 → 원본 프레임 → 화면 좌표

#### Step 1-5: 자동 촬영 시퀀스 제거

삭제: `startCaptureSequence()`, 카운트다운, CaptureOverlay, 사운드
변경: `Camera.isActive` 항상 `true`

#### Phase 1 파일 변경

```
🆕 hooks/useSegmentation.ts            — YOLOv8n-seg 로딩+추론+후처리
🆕 hooks/useSegPostprocessing.ts        — proto mask→폴리곤 로직
🆕 components/RealtimeSegOverlay.tsx    — Skia 실시간 렌더링
🆕 constants/segModel.ts               — 새 모델 상수
✏️ hooks/useObjectDetection.ts          — EfficientDet→YOLOv8n-seg 교체
✏️ hooks/useImageProcessing.ts          — 640×640 전처리 추가
✏️ components/CameraViewNoDetect.tsx    — 자동촬영 제거+마스크 오버레이
🗑️ components/CaptureOverlay.tsx        — 불필요
```

---

### Phase 2: 터치 → 서버 당도 예측 (1~2주)

**목표**: 마스크된 사과를 터치하면 서버 API로 당도 조회 (임시)

#### Step 2-1: 터치 → 사과 매핑

- Point-in-Polygon (Ray Casting) 알고리즘으로 터치 좌표가 어떤 사과 폴리곤 내부인지 판정
- Fallback: bbox 내부 체크

#### Step 2-2: 터치 → 서버 API 호출

- 현재 프레임 캡처 → 사과 bbox 크롭 → POST /predict → 당도 Tooltip 표시
- 해당 사과만 로딩 스피너 표시

#### Phase 2 파일 변경

```
🆕 hooks/useTouchToApple.ts            — point-in-polygon+터치 매핑
✏️ components/RealtimeSegOverlay.tsx    — 터치 이벤트+Tooltip
✏️ components/CameraViewNoDetect.tsx    — 터치 핸들러 연결
```

---

### Phase 3: 완전 온디바이스 당도 예측 (3~5주)

**목표**: 서버 API 완전 제거. 터치 시 온디바이스에서 즉시 당도 예측.

#### Step 3-1: EfficientNet-B0 TFLite 변환

```python
# PyTorch에서 CNN 부분만 추출 → ONNX → TFLite
class EfficientNetOnly(nn.Module):
    def __init__(self, fusion_model):
        super().__init__()
        self.cnn = fusion_model.cnn  # classifier=Identity → 1280차원 출력
    def forward(self, x):
        return self.cnn(x)

# 변환 경로: .pth → ONNX → SavedModel → TFLite (float16)
```

#### Step 3-2: MLP Head → JS 구현

```typescript
// MLP 가중치를 JSON으로 추출 → JS 행렬곱
// FC1: [128×1286] + bias[128] → ReLU
// FC2: [1×128] + bias[1] → 당도(Brix)
function mlpPredict(cnnFeatures: number[], manualFeatures: number[]): number {
  const input = [...cnnFeatures, ...manualFeatures]; // [1286]
  // FC1 + ReLU
  const hidden = new Array(128);
  for (let i = 0; i < 128; i++) {
    let sum = MLP_WEIGHTS.fc1_bias[i];
    for (let j = 0; j < 1286; j++) sum += MLP_WEIGHTS.fc1_weight[i][j] * input[j];
    hidden[i] = Math.max(0, sum);
  }
  // FC2
  let output = MLP_WEIGHTS.fc2_bias[0];
  for (let i = 0; i < 128; i++) output += MLP_WEIGHTS.fc2_weight[0][i] * hidden[i];
  return output;
}
```

#### Step 3-3: Manual Features → JS 구현

- RGB→Lab, RGB→YCbCr: 공식 기반 변환 (순수 수학)
- Scaler: `scaler.pkl`에서 mean/std 추출 → 상수
- GLCM: 0.0 고정 (서버의 cat02_first도 0.0)
- 푸른 사과 보정: Lab 색공간 조건문 그대로 포팅

#### Step 3-4: Scaler/MLP 가중치 추출 스크립트

```python
# scripts/extract_scaler.py → scalerValues.json
# scripts/extract_mlp_weights.py → mlpWeights.json (FC1:128×1286, FC2:1×128)
# scripts/convert_efficientnet_to_tflite.py → efficientnet_b0_apple_fp16.tflite
```

#### Phase 3 파일 변경

```
🆕 hooks/useSweetnessPredictor.ts      — 통합 당도 예측 파이프라인
🆕 hooks/useManualFeatures.ts          — 색공간 변환+특징 추출
🆕 constants/mlpWeights.json           — MLP 가중치 (~660KB)
🆕 constants/scalerValues.json         — Scaler mean/std
🆕 assets/efficientnet_b0_apple.tflite — CNN 모델 (~15MB)
✏️ components/RealtimeSegOverlay.tsx    — 서버→로컬 예측 교체
✏️ components/CameraViewNoDetect.tsx    — 서버 코드 제거
✏️ hooks/useTouchToApple.ts            — 서버→로컬 호출 교체
🗑️ hooks/useAnalysisApiHandler.ts      — 서버 API 상태관리 불필요
🗑️ hooks/useObjectAnalysis.ts          — 서버 fetch 불필요
🗑️ constants/api.ts                    — API 엔드포인트 불필요
🗑️ components/AnalyzedResultOverlay.tsx — RealtimeSegOverlay로 대체
```

---

## 5. 성능 목표

| 지표 | 목표 | 비고 |
|------|------|------|
| 세그멘테이션 추론 | <80ms/프레임 | GPU Delegate |
| 세그멘테이션 FPS | ≥3fps | SAMPLE_RATE 조정 |
| 당도 예측 총시간 | <500ms | 터치→결과 |
| 모델 총 크기 | <40MB | 2개 모델 합산 |
| UI 프레임레이트 | ≥30fps | 카메라 프리뷰 |
| 당도 MAE | ≤0.8 Brix | 서버 대비 약간 하락 허용 |

---

## 6. 리스크 및 대안

| 리스크 | 확률 | 대안 |
|--------|------|------|
| YOLOv8n-seg TFLite 후처리 Worklet에서 느림 | 중간 | proto 해상도 축소(160→80) 또는 bbox+타원 근사 |
| GPU Delegate 특정 연산 미지원 | 중간 | CPU fallback + NNAPI delegate |
| EfficientNet-B0 ONNX→TFLite 변환 실패 | 낮음 | `ai_edge_torch` 또는 MobileNet-V3 교체 학습 |
| GLCM JS 정확도 차이 | 높음 | 0.0 고정 (cat02_first와 동일) 또는 CNN-only 재학습 |
| 모델 2개 동시 로딩 메모리 부족 | 낮음 | 당도 모델 lazy loading + INT8 양자화 |

**최악의 경우 대안**: Manual features 제거 + CNN-only 모델 재학습 → 변환 훨씬 간단

---

## 7. 일정 추정 (1인 기준)

```
Week 1-2:  Phase 1 — TFLite 검증 + 후처리 포팅
Week 3:    Phase 1 — Skia 렌더링 + 자동촬영 제거 + 통합 테스트
Week 4:    Phase 2 — 터치 인터랙션 + 서버 당도 조회 (임시)
Week 5:    Phase 3 — 모델 변환 (EfficientNet-B0 → TFLite)
Week 6-7:  Phase 3 — JS 구현 (Manual Features + MLP + Scaler)
Week 8:    Phase 3 — 통합 + 정확도 검증 + 성능 최적화
```

**우선순위**: Phase 1 완성이 가장 중요 → Phase 2는 검증용 → Phase 3으로 서버 제거

---

## 8. 바로 시작할 첫 번째 액션

```
1. YOLOv8n-seg TFLite 텐서 shape 확인 (5분)
   → 이것이 전체 계획의 기반

2. 확인 후:
   - shape이 예상대로면 → 후처리 JS 포팅 시작
   - shape이 다르면 → ultralytics로 재변환

3. 동시에: scaler.pkl에서 mean/std 추출 (5분)
   → Phase 3 준비 선행
```

> **결론: Phase 3은 실현 가능합니다.**
> 핵심 모델(`yolov8n_seg.tflite`)이 이미 존재하고,
> FusionModel의 구조가 분리 가능하며,
> Manual Features 6개 중 5개가 단순 산술입니다.
> 8주 안에 완전 오프라인 앱을 달성할 수 있습니다.

"""
Phase 3 Step 3-1: EfficientNet-B0 TFLite 변환 스크립트

FusionModel(best_val_r2.pth)에서 CNN(EfficientNet-B0) 부분만 추출하여:
  .pth → ONNX (opset 13) → onnx2tf → TFLite (float16)

변환 경로:
  best_val_r2.pth  →  efficientnet_b0_apple.onnx
                   →  efficientnet_b0_tflite_out/
                   →  efficientnet_b0_apple_float16.tflite
                   →  FE/daldidan/assets/ 에 배치

입력: [1, 224, 224, 3] float32 (NHWC, onnx2tf가 자동 변환)
출력: [1, 1280] float32 (CNN 특징벡터)

사전 요구:
  pip install torch timm onnx onnx2tf tensorflow

사용법:
    cd scripts
    python convert_efficientnet_to_tflite.py
"""

import os
import sys
import subprocess
import shutil
import glob

import torch
import torch.nn as nn
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "BE", "ai", "services",
    "cnn_feature_seg", "outputs", "checkpoints", "best_val_r2.pth",
)
FE_ASSETS = os.path.join(PROJECT_ROOT, "FE", "daldidan", "assets")

ONNX_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b0_apple.onnx")
OUT_DIR = os.path.join(SCRIPT_DIR, "efficientnet_b0_tflite_out")

# ─────────────────────────────────────────────
# Step 0: 체크포인트 확인
# ─────────────────────────────────────────────
print("=" * 60)
print("Step 0: 체크포인트 확인")
print("=" * 60)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: 체크포인트 없음: {CHECKPOINT_PATH}")
    sys.exit(1)

print(f"체크포인트: {CHECKPOINT_PATH}")
print(f"파일 크기: {os.path.getsize(CHECKPOINT_PATH) / 1024 / 1024:.2f} MB")

# ─────────────────────────────────────────────
# Step 1: FusionModel 로드 → EfficientNet-B0 추출 → ONNX
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 1: FusionModel 로드 → EfficientNet-B0 추출 → ONNX")
print("=" * 60)

import timm

# FusionModel 정의 (원본 models/fusion_model.py 와 동일)
class FusionModel(nn.Module):
    def __init__(self, manual_feature_dim, output_dim=1):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=False)
        cnn_output_dim = self.cnn.classifier.in_features  # 1280
        self.cnn.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + manual_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, image, manual_features):
        cnn_features = self.cnn(image)
        combined = torch.cat([cnn_features, manual_features], dim=1)
        return self.fc(combined)

# CNN만 래핑하는 경량 모델
class EfficientNetOnly(nn.Module):
    """EfficientNet-B0 backbone (classifier=Identity → 1280차원 출력)"""
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)

# 로드
device = torch.device("cpu")
fusion_model = FusionModel(manual_feature_dim=6)
fusion_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
fusion_model.eval()
print("FusionModel 로드 완료")

cnn_model = EfficientNetOnly(fusion_model.cnn)
cnn_model.eval()

# 더미 추론으로 출력 확인
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = cnn_model(dummy)
print(f"PyTorch CNN 출력: shape={out.shape}, dtype={out.dtype}")
assert out.shape == (1, 1280), f"예상 (1,1280), 실제 {out.shape}"

# ONNX 변환
print(f"\nONNX 변환 중 (opset=13)...")
torch.onnx.export(
    cnn_model,
    dummy,
    ONNX_PATH,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print(f"ONNX 저장: {ONNX_PATH} ({os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB)")

# ONNX 입출력 확인
import onnx

onnx_model = onnx.load(ONNX_PATH)
print("\n--- ONNX Input Tensors ---")
for inp in onnx_model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  name: {inp.name}, shape: {shape}")

print("--- ONNX Output Tensors ---")
for o in onnx_model.graph.output:
    shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"  name: {o.name}, shape: {shape}")

del onnx_model

# ─────────────────────────────────────────────
# Step 2: onnx2tf CLI로 TFLite 변환
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: ONNX → TFLite (onnx2tf)")
print("=" * 60)

onnx2tf_bin = shutil.which("onnx2tf")
if onnx2tf_bin is None:
    venv_bin = os.path.join(SCRIPT_DIR, ".venv", "bin", "onnx2tf")
    if os.path.exists(venv_bin):
        onnx2tf_bin = venv_bin
    else:
        print("ERROR: onnx2tf를 찾을 수 없습니다. pip install onnx2tf")
        sys.exit(1)

print(f"onnx2tf 경로: {onnx2tf_bin}")

if os.path.exists(OUT_DIR):
    print(f"기존 출력 디렉토리 제거: {OUT_DIR}")
    shutil.rmtree(OUT_DIR)

# 1차 시도: 기본 옵션
cmd = [
    onnx2tf_bin,
    "-i", ONNX_PATH,
    "-o", OUT_DIR,
    "-osd",       # output saved_model directory
    "-oh5",       # output h5
    "-cotof",     # check output tensor overflows
    "-coion",     # check input/output names
]

print(f"실행: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)

if result.returncode != 0:
    print(f"onnx2tf 1차 시도 실패 (exit {result.returncode})")
    if result.stderr:
        print("STDERR (마지막 2000자):", result.stderr[-2000:])
    if result.stdout:
        print("STDOUT (마지막 2000자):", result.stdout[-2000:])

    # 2차 시도: -kat 플래그 추가
    print("\n대안 1: -kat 플래그 추가하여 재시도...")
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    cmd2 = [
        onnx2tf_bin,
        "-i", ONNX_PATH,
        "-o", OUT_DIR,
        "-osd",
        "-oh5",
        "-kat", "input",
    ]
    print(f"실행: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=SCRIPT_DIR)

    if result2.returncode != 0:
        print(f"onnx2tf 2차 시도도 실패 (exit {result2.returncode})")
        if result2.stderr:
            print("STDERR:", result2.stderr[-2000:])

        # 3차 시도: 최소 옵션
        print("\n대안 2: 최소 옵션으로 재시도...")
        if os.path.exists(OUT_DIR):
            shutil.rmtree(OUT_DIR)

        cmd3 = [
            onnx2tf_bin,
            "-i", ONNX_PATH,
            "-o", OUT_DIR,
        ]
        print(f"실행: {' '.join(cmd3)}")
        result3 = subprocess.run(cmd3, capture_output=True, text=True, cwd=SCRIPT_DIR)

        if result3.returncode != 0:
            print(f"onnx2tf 3차 시도도 실패 (exit {result3.returncode})")
            if result3.stderr:
                print("STDERR:", result3.stderr[-2000:])
            if result3.stdout:
                print("STDOUT:", result3.stdout[-2000:])
            print("\n❌ onnx2tf 변환 실패. 수동 확인 필요.")
            sys.exit(1)
        else:
            print("onnx2tf 3차 시도 성공! (최소 옵션)")
    else:
        print("onnx2tf 2차 시도 성공! (-kat 옵션)")
else:
    print("onnx2tf 변환 성공!")

if result.stdout:
    print("STDOUT (마지막 500자):", result.stdout[-500:])

# ─────────────────────────────────────────────
# Step 3: TFLite 파일 찾기 + 텐서 확인
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: TFLite 텐서 shape 확인")
print("=" * 60)

import tensorflow as tf
import numpy as np

tflite_files = glob.glob(os.path.join(OUT_DIR, "**/*.tflite"), recursive=True)
if not tflite_files:
    print("WARNING: TFLite 파일을 찾을 수 없습니다. SavedModel에서 직접 변환 시도...")

    # SavedModel → TFLite 직접 변환
    saved_model_dir = OUT_DIR
    if not os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
        # 서브디렉토리 탐색
        for d in os.listdir(OUT_DIR):
            candidate = os.path.join(OUT_DIR, d)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "saved_model.pb")):
                saved_model_dir = candidate
                break

    if os.path.exists(os.path.join(saved_model_dir, "saved_model.pb")):
        print(f"SavedModel 발견: {saved_model_dir}")

        # float32 변환
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        float32_path = os.path.join(OUT_DIR, "efficientnet_b0_apple_float32.tflite")
        with open(float32_path, "wb") as f:
            f.write(tflite_model)
        print(f"float32 TFLite 저장: {float32_path} ({len(tflite_model) / 1024 / 1024:.2f} MB)")

        # float16 변환
        converter16 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter16.target_spec.supported_types = [tf.float16]
        tflite_model_fp16 = converter16.convert()
        float16_path = os.path.join(OUT_DIR, "efficientnet_b0_apple_float16.tflite")
        with open(float16_path, "wb") as f:
            f.write(tflite_model_fp16)
        print(f"float16 TFLite 저장: {float16_path} ({len(tflite_model_fp16) / 1024 / 1024:.2f} MB)")

        tflite_files = [float16_path, float32_path]
    else:
        print(f"ERROR: SavedModel을 찾을 수 없습니다: {saved_model_dir}")
        print("디렉토리 내용:")
        for root, dirs, files in os.walk(OUT_DIR):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        sys.exit(1)

# float16 버전 우선 사용
tflite_path = tflite_files[0]
for f in tflite_files:
    if "float16" in f:
        tflite_path = f
        break

print(f"\n선택된 TFLite: {tflite_path}")
print(f"파일 크기: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")

# 텐서 shape 확인
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

print("\n--- Input Tensors ---")
for t in interpreter.get_input_details():
    print(f"  name: {t['name']}")
    print(f"  shape: {t['shape']}")
    print(f"  dtype: {t['dtype']}")

print("\n--- Output Tensors ---")
for i, t in enumerate(interpreter.get_output_details()):
    print(f"  [{i}] name: {t['name']}")
    print(f"      shape: {t['shape']}")
    print(f"      dtype: {t['dtype']}")

# 더미 추론 테스트
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# ImageNet 정규화된 더미 입력 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
dummy_input = np.random.rand(*input_shape).astype(np.float32)
# NCHW → NHWC 변환이 필요한 경우 확인
if input_shape[-1] == 3:
    print("\n입력 형식: NHWC (채널 마지막)")
elif input_shape[1] == 3:
    print("\n입력 형식: NCHW (채널 우선)")

dummy_input = dummy_input.astype(input_dtype)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

print("\n--- 더미 추론 결과 ---")
for i, t in enumerate(output_details):
    output = interpreter.get_tensor(t['index'])
    print(f"  Output[{i}]: shape={output.shape}, dtype={output.dtype}, "
          f"min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

# 출력 shape 검증
expected_output_dim = 1280
output_shape = interpreter.get_tensor(output_details[0]['index']).shape
if output_shape[-1] == expected_output_dim or (len(output_shape) == 2 and output_shape[1] == expected_output_dim):
    print(f"\n✅ 출력 차원 검증 성공: {expected_output_dim}차원 특징 벡터")
else:
    print(f"\n⚠️ 출력 shape가 예상과 다름: {output_shape} (예상: [1, {expected_output_dim}])")
    print("후처리 코드에서 shape 조정이 필요할 수 있습니다.")

# ─────────────────────────────────────────────
# Step 4: FE assets로 복사
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: FE assets로 복사")
print("=" * 60)

dest_path = os.path.join(FE_ASSETS, "efficientnet_b0_apple.tflite")
shutil.copy2(tflite_path, dest_path)
print(f"복사 완료: {dest_path}")
print(f"파일 크기: {os.path.getsize(dest_path) / 1024 / 1024:.2f} MB")

# float32 버전도 복사 (디버깅용)
for f in tflite_files:
    if "float32" in f:
        dest32 = os.path.join(FE_ASSETS, "efficientnet_b0_apple_float32.tflite")
        shutil.copy2(f, dest32)
        print(f"float32 버전 복사: {dest32} ({os.path.getsize(dest32) / 1024 / 1024:.2f} MB)")
        break

# ─────────────────────────────────────────────
# Step 5: 요약
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ EfficientNet-B0 TFLite 변환 완료!")
print("=" * 60)
print(f"""
변환 결과:
  ONNX 원본:     {os.path.getsize(ONNX_PATH) / 1024 / 1024:.2f} MB
  TFLite (fp16): {os.path.getsize(dest_path) / 1024 / 1024:.2f} MB
  입력 텐서:     {list(input_shape)}
  출력 텐서:     {list(output_shape)}
  배치 경로:     {dest_path}

다음 단계:
  1. scripts/extract_scaler.py    → scalerValues.json 생성
  2. scripts/extract_mlp_weights.py → mlpWeights.json 생성
  3. FE에서 TFLite 모델 로딩 + 추론 구현
""")

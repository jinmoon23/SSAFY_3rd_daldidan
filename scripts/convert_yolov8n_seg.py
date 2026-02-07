"""
YOLOv8n-seg → TFLite 변환 및 텐서 shape 검증 스크립트
Phase 3 Step 1-1: 모델 변환 + 입출력 텐서 구조 확인

전략: ultralytics의 built-in export 대신, ONNX(opset 13) → onnx2tf CLI 직접 호출
"""

import os
import sys
import subprocess
import shutil
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_BIN = os.path.join(SCRIPT_DIR, ".venv", "bin")
FE_ASSETS = os.path.join(SCRIPT_DIR, "..", "FE", "daldidan", "assets")

# ─────────────────────────────────────────────
# Step 1: YOLOv8n-seg 다운로드 + ONNX 변환
# ─────────────────────────────────────────────
print("=" * 60)
print("Step 1: YOLOv8n-seg → ONNX (opset 13)")
print("=" * 60)

import torch
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
print(f"모델 로드 완료: {model.model_name}")

# ONNX로 먼저 변환 (opset 13 — onnx2tf 호환성 높음)
onnx_path = model.export(format="onnx", imgsz=640, opset=13, simplify=True)
print(f"ONNX 변환 완료: {onnx_path}")

# ─────────────────────────────────────────────
# Step 2: onnx2tf CLI로 TFLite 변환
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: ONNX → TFLite (onnx2tf)")
print("=" * 60)

onnx2tf_bin = os.path.join(VENV_BIN, "onnx2tf")
out_dir = os.path.join(SCRIPT_DIR, "yolov8n_seg_tflite_out")

cmd = [
    onnx2tf_bin,
    "-i", onnx_path,
    "-o", out_dir,
    "-osd",       # output saved_model directory
    "-cotof",     # check output tensor overflows
    "-coion",     # check input/output names
]

print(f"실행: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)

if result.returncode != 0:
    print(f"onnx2tf 실패 (exit {result.returncode})")
    print("STDERR:", result.stderr[-2000:] if result.stderr else "없음")
    print("\n대안: onnx2tf에 -kat 플래그 추가하여 재시도...")

    cmd2 = [
        onnx2tf_bin,
        "-i", onnx_path,
        "-o", out_dir,
        "-osd",
        "-kat", "input",
    ]
    print(f"실행: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=SCRIPT_DIR)

    if result2.returncode != 0:
        print(f"onnx2tf 재시도도 실패")
        print("STDERR:", result2.stderr[-2000:] if result2.stderr else "없음")

        # 최종 대안: SavedModel에서 직접 TFLite 변환
        print("\n최종 대안: torch → TF SavedModel → TFLite 직접 변환 시도...")
        
        # PyTorch 모델의 출력 shape만 확인 (TFLite 변환 실패해도 유용한 정보)
        print("\n" + "=" * 60)
        print("PyTorch 모델 출력 shape 확인 (TFLite 없이)")
        print("=" * 60)
        import numpy as np
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            pt_output = model.model(dummy)
        if isinstance(pt_output, (list, tuple)):
            for i, o in enumerate(pt_output):
                if isinstance(o, torch.Tensor):
                    print(f"  Output[{i}]: shape={o.shape}, dtype={o.dtype}")
                elif isinstance(o, (list, tuple)):
                    for j, oo in enumerate(o):
                        if isinstance(oo, torch.Tensor):
                            print(f"  Output[{i}][{j}]: shape={oo.shape}, dtype={oo.dtype}")
        else:
            print(f"  Output: shape={pt_output.shape}, dtype={pt_output.dtype}")
        sys.exit(1)
else:
    print("onnx2tf 변환 성공!")

# ─────────────────────────────────────────────
# Step 3: TFLite 파일 찾기 + 텐서 확인
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: TFLite 텐서 shape 확인")
print("=" * 60)

import tensorflow as tf
import numpy as np

tflite_files = glob.glob(os.path.join(out_dir, "**/*.tflite"), recursive=True)
if not tflite_files:
    print("ERROR: TFLite 파일을 찾을 수 없습니다")
    sys.exit(1)

# float32 버전 우선
tflite_path = tflite_files[0]
for f in tflite_files:
    if "float32" in f:
        tflite_path = f
        break

print(f"TFLite 파일: {tflite_path}")
print(f"파일 크기: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")

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

# 더미 추론
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
dummy_input = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

print("\n--- 더미 추론 결과 ---")
for i, t in enumerate(output_details):
    output = interpreter.get_tensor(t['index'])
    print(f"  Output[{i}]: shape={output.shape}, dtype={output.dtype}, "
          f"min={output.min():.4f}, max={output.max():.4f}")

# ─────────────────────────────────────────────
# Step 4: FE assets로 복사
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: FE assets로 복사")
print("=" * 60)

dest_path = os.path.join(FE_ASSETS, "yolov8n_seg.tflite")
shutil.copy2(tflite_path, dest_path)
print(f"복사 완료: {dest_path}")
print(f"파일 크기: {os.path.getsize(dest_path) / 1024 / 1024:.2f} MB")

print("\n✅ 모든 단계 완료!")

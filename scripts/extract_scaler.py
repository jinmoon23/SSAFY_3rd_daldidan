"""
Phase 3 Step 3-4: Scaler mean/std 추출 스크립트

scaler.pkl (StandardScaler)에서 mean, std(scale)를 JSON으로 추출.
Manual Features 6차원: [Rn, C, ycbcr_diff, ycbcr_norm, cat02_first, cluster_shadow]

출력: scripts/scalerValues.json → FE/daldidan/constants/scalerValues.json 에 배치

사용법:
    cd scripts
    python extract_scaler.py
"""

import os
import sys
import json

import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
SCALER_PATH = os.path.join(
    PROJECT_ROOT, "BE", "ai", "services",
    "cnn_feature_seg", "outputs", "checkpoints", "scaler.pkl",
)
FE_CONSTANTS = os.path.join(PROJECT_ROOT, "FE", "daldidan", "constants")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "scalerValues.json")
FE_OUTPUT_PATH = os.path.join(FE_CONSTANTS, "scalerValues.json")

# ─────────────────────────────────────────────
# Step 0: scaler.pkl 확인
# ─────────────────────────────────────────────
print("=" * 60)
print("Scaler 추출: scaler.pkl → scalerValues.json")
print("=" * 60)

if not os.path.exists(SCALER_PATH):
    print(f"ERROR: scaler.pkl 없음: {SCALER_PATH}")
    sys.exit(1)

print(f"scaler.pkl: {SCALER_PATH}")

# ─────────────────────────────────────────────
# Step 1: Scaler 로드 + 값 추출
# ─────────────────────────────────────────────
scaler = joblib.load(SCALER_PATH)

print(f"\nScaler 타입: {type(scaler).__name__}")
print(f"n_features: {scaler.n_features_in_}")

mean = scaler.mean_       # [6]
scale = scaler.scale_     # [6] (= std)
var = scaler.var_         # [6]

feature_names = ["Rn", "C", "ycbcr_diff", "ycbcr_norm", "cat02_first", "cluster_shadow"]

print(f"\n--- Scaler 값 ---")
print(f"{'Feature':<20} {'mean':>12} {'std(scale)':>12} {'var':>12}")
print("-" * 60)
for i, name in enumerate(feature_names):
    print(f"{name:<20} {mean[i]:>12.6f} {scale[i]:>12.6f} {var[i]:>12.6f}")

# ─────────────────────────────────────────────
# Step 2: JSON 저장
# ─────────────────────────────────────────────
scaler_values = {
    "mean": mean.tolist(),      # [6]
    "scale": scale.tolist(),    # [6] (std)
    "feature_names": feature_names,
    "n_features": int(scaler.n_features_in_),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(scaler_values, f, indent=2)

file_size = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\n저장: {OUTPUT_PATH} ({file_size:.1f} KB)")

# FE constants 디렉토리에도 복사
os.makedirs(FE_CONSTANTS, exist_ok=True)
with open(FE_OUTPUT_PATH, "w") as f:
    json.dump(scaler_values, f, indent=2)
print(f"복사: {FE_OUTPUT_PATH} ({file_size:.1f} KB)")

# ─────────────────────────────────────────────
# Step 3: 검증 (더미 정규화 비교)
# ─────────────────────────────────────────────
print("\n--- 검증: sklearn vs JSON 정규화 비교 ---")

dummy_features = np.array([0.35, 0.6, -10.0, 0.48, 0.0, 3.5])
print(f"입력: {dummy_features}")

# sklearn 정규화
sklearn_result = scaler.transform([dummy_features])[0]

# JSON 값으로 수동 정규화 (JS에서 실행될 로직)
json_result = (dummy_features - mean) / scale

print(f"sklearn: {sklearn_result}")
print(f"JSON:    {json_result}")
print(f"차이:    {np.abs(sklearn_result - json_result).max():.10f}")

if np.allclose(sklearn_result, json_result):
    print("\n✅ Scaler 추출 검증 성공!")
else:
    print("\n⚠️ 수치 불일치. 확인 필요.")

print(f"""
JS에서의 사용법:
  import scalerValues from '../constants/scalerValues.json';
  
  function scaleFeatures(features: number[]): number[] {{
    return features.map((x, i) => (x - scalerValues.mean[i]) / scalerValues.scale[i]);
  }}
""")

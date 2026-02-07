"""
Phase 3 Step 3-4: MLP 가중치 추출 스크립트

FusionModel(best_val_r2.pth)에서 MLP Head 가중치를 JSON으로 추출.
FC1: [128 × 1286] + bias[128] → ReLU
FC2: [1 × 128]   + bias[1]   → 당도(Brix)

출력: scripts/mlpWeights.json → FE/daldidan/constants/mlpWeights.json 에 배치

사용법:
    cd scripts
    python extract_mlp_weights.py
"""

import os
import sys
import json

import torch
import torch.nn as nn
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "BE", "ai", "services",
    "cnn_feature_seg", "outputs", "checkpoints", "best_val_r2.pth",
)
FE_CONSTANTS = os.path.join(PROJECT_ROOT, "FE", "daldidan", "constants")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "mlpWeights.json")
FE_OUTPUT_PATH = os.path.join(FE_CONSTANTS, "mlpWeights.json")

# ─────────────────────────────────────────────
# Step 0: 체크포인트 확인
# ─────────────────────────────────────────────
print("=" * 60)
print("MLP 가중치 추출: best_val_r2.pth → mlpWeights.json")
print("=" * 60)

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: 체크포인트 없음: {CHECKPOINT_PATH}")
    sys.exit(1)

# ─────────────────────────────────────────────
# Step 1: FusionModel 로드
# ─────────────────────────────────────────────
import timm

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

device = torch.device("cpu")
model = FusionModel(manual_feature_dim=6)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("FusionModel 로드 완료")

# ─────────────────────────────────────────────
# Step 2: MLP 가중치 추출
# ─────────────────────────────────────────────
print("\n--- MLP (model.fc) 구조 ---")
for name, param in model.fc.named_parameters():
    print(f"  {name}: shape={list(param.shape)}, dtype={param.dtype}")

# fc = Sequential(
#   (0): Linear(1286, 128)  → weight [128, 1286], bias [128]
#   (1): ReLU()
#   (2): Linear(128, 1)     → weight [1, 128], bias [1]
# )
fc1_weight = model.fc[0].weight.detach().numpy()  # [128, 1286]
fc1_bias = model.fc[0].bias.detach().numpy()       # [128]
fc2_weight = model.fc[2].weight.detach().numpy()   # [1, 128]
fc2_bias = model.fc[2].bias.detach().numpy()       # [1]

print(f"\nfc1_weight: {fc1_weight.shape}")
print(f"fc1_bias:   {fc1_bias.shape}")
print(f"fc2_weight: {fc2_weight.shape}")
print(f"fc2_bias:   {fc2_bias.shape}")

# ─────────────────────────────────────────────
# Step 3: JSON 저장
# ─────────────────────────────────────────────
weights = {
    "fc1_weight": fc1_weight.tolist(),  # [128][1286]
    "fc1_bias": fc1_bias.tolist(),      # [128]
    "fc2_weight": fc2_weight.tolist(),  # [1][128]
    "fc2_bias": fc2_bias.tolist(),      # [1]
    "_meta": {
        "source": "best_val_r2.pth",
        "fc1": "Linear(1286, 128) → ReLU",
        "fc2": "Linear(128, 1) → Brix",
        "input_order": "cnn_features[1280] + manual_features[6]",
        "manual_features": ["Rn", "C", "ycbcr_diff", "ycbcr_norm", "cat02_first", "cluster_shadow"],
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(weights, f)

file_size = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\n저장: {OUTPUT_PATH} ({file_size:.1f} KB)")

# FE constants 디렉토리에도 복사
os.makedirs(FE_CONSTANTS, exist_ok=True)
with open(FE_OUTPUT_PATH, "w") as f:
    json.dump(weights, f)
print(f"복사: {FE_OUTPUT_PATH} ({file_size:.1f} KB)")

# ─────────────────────────────────────────────
# Step 4: 검증 (더미 추론 비교)
# ─────────────────────────────────────────────
print("\n--- 검증: PyTorch vs JSON 가중치 추론 비교 ---")

# PyTorch 추론
dummy_cnn = torch.randn(1, 1280)
dummy_manual = torch.randn(1, 6)
dummy_combined = torch.cat([dummy_cnn, dummy_manual], dim=1)

with torch.no_grad():
    pt_output = model.fc(dummy_combined).item()

# JSON 가중치 추론 (JS에서 실행될 로직과 동일)
combined_np = dummy_combined.numpy().flatten()  # [1286]

hidden = np.zeros(128)
for i in range(128):
    s = fc1_bias[i]
    for j in range(1286):
        s += fc1_weight[i][j] * combined_np[j]
    hidden[i] = max(0, s)  # ReLU

output = fc2_bias[0]
for i in range(128):
    output += fc2_weight[0][i] * hidden[i]

print(f"  PyTorch 출력: {pt_output:.6f}")
print(f"  JSON 추론:    {output:.6f}")
print(f"  차이:         {abs(pt_output - output):.8f}")

if abs(pt_output - output) < 1e-4:
    print("\n✅ MLP 가중치 추출 검증 성공!")
else:
    print("\n⚠️ 수치 차이가 큼. float 정밀도 확인 필요.")

print(f"""
다음 단계:
  FE에서 mlpWeights.json을 import하여 JS 행렬곱 구현
  예: import mlpWeights from '../constants/mlpWeights.json';
""")

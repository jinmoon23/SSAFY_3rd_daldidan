#!/usr/bin/env python3
"""
Phase 3: FusionModel에서 온디바이스용 자산 추출

1. scaler.pkl → scalerValues.json (mean, scale 6차원)
2. best_val_r2.pth → mlpWeights.json (FC1, FC2 가중치+바이어스)
3. best_val_r2.pth → efficientnet_b0_apple.tflite (CNN 부분만)

Usage:
  pip install torch timm joblib onnx onnx-tf tensorflow
  python extract_sweetness_model.py
"""

import os
import sys
import json
import numpy as np
import torch
import joblib

# === 경로 설정 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # FE/daldidan
BE_MODEL_DIR = os.path.join(
    SCRIPT_DIR, '..', '..', '..', 'BE', 'ai', 'services', 'cnn_feature_seg'
)
CHECKPOINT_DIR = os.path.join(BE_MODEL_DIR, 'outputs', 'checkpoints')
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_val_r2.pth')
SCALER_PATH = os.path.join(CHECKPOINT_DIR, 'scaler.pkl')

# 출력 경로
ASSETS_DIR = os.path.join(PROJECT_DIR, 'assets')
CONSTANTS_DIR = os.path.join(PROJECT_DIR, 'constants')
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(CONSTANTS_DIR, exist_ok=True)

SCALER_JSON_PATH = os.path.join(CONSTANTS_DIR, 'scalerValues.json')
MLP_JSON_PATH = os.path.join(CONSTANTS_DIR, 'mlpWeights.json')
TFLITE_PATH = os.path.join(ASSETS_DIR, 'efficientnet_b0_apple.tflite')
ONNX_PATH = os.path.join(ASSETS_DIR, 'efficientnet_b0_apple.onnx')

MANUAL_FEATURE_DIM = 6


def check_files():
    """필수 파일 존재 확인"""
    for path, name in [(MODEL_PATH, 'best_val_r2.pth'), (SCALER_PATH, 'scaler.pkl')]:
        if not os.path.exists(path):
            print(f"❌ {name} not found at: {path}")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✅ {name} found ({size_mb:.2f} MB)")


def extract_scaler():
    """scaler.pkl → scalerValues.json"""
    print("\n=== Step 1: Scaler 추출 ===")
    scaler = joblib.load(SCALER_PATH)

    scaler_data = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'feature_names': ['Rn', 'C', 'ycbcr_diff', 'ycbcr_norm', 'cat02_first', 'cluster_shadow'],
        'n_features': MANUAL_FEATURE_DIM,
    }

    with open(SCALER_JSON_PATH, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    print(f"  mean:  {scaler_data['mean']}")
    print(f"  scale: {scaler_data['scale']}")
    print(f"  ✅ Saved: {SCALER_JSON_PATH}")
    return scaler_data


def extract_mlp_weights():
    """best_val_r2.pth → mlpWeights.json"""
    print("\n=== Step 2: MLP 가중치 추출 ===")

    state_dict = torch.load(MODEL_PATH, map_location='cpu')

    # FusionModel.fc = Sequential(Linear(1280+6, 128), ReLU, Linear(128, 1))
    fc0_weight = state_dict['fc.0.weight']  # [128, 1286]
    fc0_bias = state_dict['fc.0.bias']      # [128]
    fc2_weight = state_dict['fc.2.weight']  # [1, 128]
    fc2_bias = state_dict['fc.2.bias']      # [1]

    mlp_data = {
        'fc1': {
            'weight': fc0_weight.numpy().tolist(),  # [128, 1286]
            'bias': fc0_bias.numpy().tolist(),       # [128]
        },
        'fc2': {
            'weight': fc2_weight.numpy().tolist(),  # [1, 128]
            'bias': fc2_bias.numpy().tolist(),       # [1]
        },
        'cnn_output_dim': fc0_weight.shape[1] - MANUAL_FEATURE_DIM,  # 1280
        'manual_feature_dim': MANUAL_FEATURE_DIM,
        'hidden_dim': 128,
    }

    print(f"  FC1: weight {list(fc0_weight.shape)}, bias {list(fc0_bias.shape)}")
    print(f"  FC2: weight {list(fc2_weight.shape)}, bias {list(fc2_bias.shape)}")
    print(f"  CNN output dim: {mlp_data['cnn_output_dim']}")

    with open(MLP_JSON_PATH, 'w') as f:
        json.dump(mlp_data, f)
    size_mb = os.path.getsize(MLP_JSON_PATH) / (1024 * 1024)
    print(f"  ✅ Saved: {MLP_JSON_PATH} ({size_mb:.2f} MB)")
    return mlp_data


def convert_cnn_to_tflite():
    """EfficientNet-B0 CNN 부분만 TFLite로 변환"""
    print("\n=== Step 3: EfficientNet-B0 → TFLite 변환 ===")

    import timm

    # FusionModel의 CNN 부분만 재구성
    class CNNOnly(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.cnn = timm.create_model('efficientnet_b0', pretrained=False)
            cnn_output_dim = self.cnn.classifier.in_features
            self.cnn.classifier = torch.nn.Identity()

            # CNN 가중치만 로드
            cnn_state = {
                k.replace('cnn.', ''): v
                for k, v in state_dict.items()
                if k.startswith('cnn.')
            }
            self.cnn.load_state_dict(cnn_state, strict=True)
            print(f"  CNN weights loaded ({len(cnn_state)} tensors)")
            print(f"  CNN output dim: {cnn_output_dim}")

        def forward(self, x):
            return self.cnn(x)

    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    cnn_model = CNNOnly(state_dict)
    cnn_model.eval()

    # Step 3a: PyTorch → ONNX
    print("  3a. Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        test_output = cnn_model(dummy_input)
        print(f"  Test output shape: {test_output.shape}")

    torch.onnx.export(
        cnn_model,
        dummy_input,
        ONNX_PATH,
        opset_version=13,
        input_names=['image'],
        output_names=['cnn_features'],
        dynamic_axes=None,
    )
    onnx_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  ✅ ONNX saved: {ONNX_PATH} ({onnx_size:.2f} MB)")

    # Step 3b: ONNX → TFLite (onnx2tf 사용)
    print("  3b. Converting ONNX → TFLite...")
    try:
        import onnx
        import onnx_tf
        import tensorflow as tf

        # ONNX → TF SavedModel
        onnx_model = onnx.load(ONNX_PATH)
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        saved_model_dir = os.path.join(ASSETS_DIR, 'efficientnet_b0_saved_model')
        tf_rep.export_graph(saved_model_dir)

        # TF SavedModel → TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = []  # float32 유지
        tflite_model = converter.convert()

        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)

        tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
        print(f"  ✅ TFLite saved: {TFLITE_PATH} ({tflite_size:.2f} MB)")

        # 정리
        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        os.remove(ONNX_PATH)
        print("  ✅ Cleaned up intermediate files")

    except ImportError as e:
        print(f"\n  ⚠️ TFLite 변환 라이브러리 부족: {e}")
        print(f"  ONNX 파일은 저장됨: {ONNX_PATH}")
        print(f"  수동 변환: pip install onnx onnx-tf tensorflow 후 재실행")
        print(f"  또는 onnx2tf CLI: onnx2tf -i {ONNX_PATH} -o {ASSETS_DIR}")


def verify_tflite():
    """TFLite 모델 검증"""
    if not os.path.exists(TFLITE_PATH):
        print("\n⚠️ TFLite 파일 미존재 — 검증 스킵")
        return

    print("\n=== Step 4: TFLite 검증 ===")
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"  Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
        print(f"  Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

        # 더미 추론
        dummy = np.random.randn(*input_details[0]['shape']).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"  Dummy inference output: {output.flatten()[:5]}")
        print("  ✅ TFLite 검증 완료")
    except ImportError:
        print("  ⚠️ tensorflow 미설치 — 검증 스킵")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3: FusionModel → 온디바이스 자산 추출")
    print("=" * 60)

    check_files()
    extract_scaler()
    extract_mlp_weights()
    convert_cnn_to_tflite()
    verify_tflite()

    print("\n" + "=" * 60)
    print("✅ 추출 완료! 생성된 파일:")
    for p in [SCALER_JSON_PATH, MLP_JSON_PATH, TFLITE_PATH, ONNX_PATH]:
        if os.path.exists(p):
            size = os.path.getsize(p) / (1024 * 1024)
            print(f"  {os.path.relpath(p, PROJECT_DIR)} ({size:.2f} MB)")
    print("=" * 60)

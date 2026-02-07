#!/bin/bash
# ============================================
# iOS 개발 환경 설정 스크립트
# Xcode 설치 후 이 스크립트를 실행하세요
# Usage: bash scripts/setup-ios.sh
# ============================================

set -e

echo "🍎 iOS 개발 환경 설정을 시작합니다..."

# 1. Xcode Command Line Tools 확인
echo ""
echo "📌 [1/5] Xcode 확인..."
if ! xcode-select -p &>/dev/null; then
  echo "❌ Xcode가 설치되지 않았습니다. App Store에서 Xcode를 먼저 설치해주세요."
  exit 1
fi

XCODE_PATH=$(xcode-select -p)
echo "  현재 developer directory: $XCODE_PATH"

if [[ "$XCODE_PATH" == "/Library/Developer/CommandLineTools" ]]; then
  echo "  ⚠️  developer directory가 Command Line Tools로 설정되어 있습니다."
  echo "  Xcode.app으로 변경합니다..."
  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
  echo "  ✅ developer directory 변경 완료"
fi

# 2. Xcode 라이선스 동의
echo ""
echo "📌 [2/5] Xcode 라이선스 확인..."
if ! sudo xcodebuild -license check &>/dev/null; then
  echo "  Xcode 라이선스에 동의합니다..."
  sudo xcodebuild -license accept
fi
echo "  ✅ Xcode 라이선스 동의 완료"

# 3. iOS Simulator 확인
echo ""
echo "📌 [3/5] iOS Simulator 런타임 확인..."
xcrun simctl list runtimes | grep -i ios || echo "  ⚠️  iOS 시뮬레이터 런타임이 없습니다. Xcode > Settings > Platforms에서 설치해주세요."

# 4. CocoaPods 확인
echo ""
echo "📌 [4/5] CocoaPods 확인..."
if ! command -v pod &>/dev/null; then
  echo "  CocoaPods 설치 중..."
  brew install cocoapods
fi
echo "  ✅ CocoaPods: $(pod --version)"

# 5. Pod install
echo ""
echo "📌 [5/5] Pod install 실행..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/ios"

pod install

echo ""
echo "============================================"
echo "✅ iOS 개발 환경 설정이 완료되었습니다!"
echo ""
echo "다음 명령으로 iOS 앱을 실행하세요:"
echo "  npx expo run:ios"
echo ""
echo "또는 Xcode에서 직접 열기:"
echo "  open ios/DaldidanDev.xcworkspace"
echo "============================================"

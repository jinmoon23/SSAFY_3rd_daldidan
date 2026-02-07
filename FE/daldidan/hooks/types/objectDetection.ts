// daldidan/hooks/types/objectDetection.ts

// Vision Camera 프레임 프로세서에서 사용되는 탐지 결과 타입
export interface Detection {
  x: number; // Vision Camera frame processor 좌표계 기준 (모델 입력과 다를 수 있음)
  y: number;
  width: number;
  height: number;
  score?: number;
  class_id?: number;
  // sugar_content 필드는 이제 API 응답에서 오므로 여기서는 제거하거나 옵션으로 둡니다.
  // sugar_content?: number;
}

// 백엔드 API에서 받아올 객체 탐지 및 분석 결과의 타입 정의
// 이 타입은 useObjectAnalysis, useAnalysisApiHandler, CameraViewNoDetect 등에서 사용됩니다.
export interface AnalyzedObjectResult {
  id?: number; // 백엔드에서 부여한 고유 ID (응답 예시에는 id가 있네요)
  sugar_content?: number | null; // 당도 값 (nullable)
  bbox: {
    // 바운딩 박스 좌표 구조
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
  segmentation?: {
    points: number[][];
  } | null;
  // ... 기타 필드 // 응답 예시에 segmentation이 있네요 (null 또는 다른 형태일 수 있습니다)
  // TODO: 백엔드 응답에 'class_id'나 'label', 'score' 등이 직접 포함되어 있는지 확인 필요
  // 현재 예시 응답에는 없지만, 객체의 종류를 구분하려면 이 정보가 필요합니다.
  // 만약 백엔드가 id만 반환한다면, 이 id를 프론트엔드에서 클래스/라벨에 매핑하는 로직이 필요할 수 있습니다.
  // 여기서는 'label'과 'class_id' 필드가 없다고 가정하고 일단 진행합니다.
  // 만약 있다면 여기에 추가해야 합니다: class_id?: number; label?: string; score?: number;
}

// API 응답이 분석된 객체 결과들의 배열이라고 가정
export interface ScreenshotAnalysisResponse {
  results: AnalyzedObjectResult[]; // 백엔드 응답의 "results" 키 아래에 객체 배열이 담김
}

// Phase 3: 온디바이스 실시간 세그멘테이션 결과
export interface SegmentationResult {
  id: number;
  bbox: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
  polygon: number[][]; // 세그멘테이션 폴리곤 좌표 [[x,y], ...]
  score: number;
  sweetness?: number; // 터치 후 채워지는 당도 (Phase 2~3)
  isLoading?: boolean; // 당도 예측 중
}

// 크롭된 이미지 데이터 (Worklet에서 JS로 전달)
export interface CroppedImageData {
  data: number[];
  width: number;
  height: number;
  isJPEG: boolean;
}

// DetectionResult (기존 호환)
export interface DetectionResult {
  detection: Detection;
  timestamp: number;
}

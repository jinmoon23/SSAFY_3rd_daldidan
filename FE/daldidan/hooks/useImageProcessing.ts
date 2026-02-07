import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { CroppedImageData, Detection } from './types/objectDetection';

export const useImageProcessing = () => {
  const { resize } = useResizePlugin();

  const logWorklet = Worklets.createRunOnJS((message: string) => {
    console.log(message);
  });

  const clamp = (value: number, min: number, max: number) => {
    'worklet';
    return Math.max(min, Math.min(value, max));
  };

  const preprocessFrame = (frame: any, targetSize: number) => {
    'worklet';
     const isPortrait = frame.height > frame.width;
  const shortSide = Math.min(frame.width, frame.height);
  const cropX = (frame.width - shortSide) / 2;
  const cropY = (frame.height - shortSide) / 2;

  return resize(frame, {
    scale: { width: targetSize, height: targetSize },
    pixelFormat: 'rgb',
    dataType: 'uint8',
    crop: {
      x: cropX,
      y: cropY,
      width: shortSide,
      height: shortSide,
    },
  });
};

  const extractCroppedData = async (
    frame: any,
    detection: Detection
  ): Promise<CroppedImageData | null> => {
    'worklet';
    try {
      const { x, y, width, height } = detection;

      // 크롭할 영역이 프레임을 벗어나지 않도록 보정
      const safeX = Math.max(0, Math.min(x, frame.width - 1));
      const safeY = Math.max(0, Math.min(y, frame.height - 1));
      const safeWidth = Math.min(width, frame.width - safeX);
      const safeHeight = Math.min(height, frame.height - safeY);

      // 최소 크기 제한을 더 크게 설정
      if (safeWidth < 20 || safeHeight < 20) {
        // logWorklet(`[Worklet] Area too small: ${safeWidth}x${safeHeight}`);
        return null;
      }

      // 최대 크기 제한을 더 작게 설정
      const maxSize = 100;
      const resized = resize(frame, {
        scale: {
          width: Math.min(safeWidth, maxSize),
          height: Math.min(safeHeight, maxSize),
        },
        pixelFormat: 'rgb',
        dataType: 'uint8',
        crop: {
          x: safeX,
          y: safeY,
          width: safeWidth,
          height: safeHeight,
        },
      });

      if (!resized) {
        logWorklet('[Worklet] Resize operation failed');
        return null;
      }

      const dataArray = Array.from(new Uint8Array(resized));
      // logWorklet(`[Worklet] Extracted data size: ${dataArray.length} bytes`);

      return {
        data: dataArray,
        width: Math.min(safeWidth, maxSize),
        height: Math.min(safeHeight, maxSize),
        isJPEG: false,
      };
    } catch (error) {
      logWorklet(`[Worklet] Extraction error: ${error}`);
      return null;
    }
  };

  // YOLOv8n-seg용 전처리: float32 정규화 (0~1), 640×640
  const preprocessFrameForSeg = (frame: any, targetSize: number) => {
    'worklet';
    const shortSide = Math.min(frame.width, frame.height);
    const cropX = (frame.width - shortSide) / 2;
    const cropY = (frame.height - shortSide) / 2;

    return resize(frame, {
      scale: { width: targetSize, height: targetSize },
      pixelFormat: 'rgb',
      dataType: 'float32',
      crop: {
        x: cropX,
        y: cropY,
        width: shortSide,
        height: shortSide,
      },
    });
  };

  // Phase 3: bbox 크롭 → targetSize×targetSize float32 (0~1)
  const cropAndResize = (
    frame: any,
    cropX: number,
    cropY: number,
    cropW: number,
    cropH: number,
    targetSize: number
  ) => {
    'worklet';
    return resize(frame, {
      scale: { width: targetSize, height: targetSize },
      pixelFormat: 'rgb',
      dataType: 'float32',
      crop: { x: cropX, y: cropY, width: cropW, height: cropH },
    });
  };

  // Phase 3: bbox 크롭 → targetSize×targetSize uint8
  const cropAndResizeUint8 = (
    frame: any,
    cropX: number,
    cropY: number,
    cropW: number,
    cropH: number,
    targetSize: number
  ) => {
    'worklet';
    return resize(frame, {
      scale: { width: targetSize, height: targetSize },
      pixelFormat: 'rgb',
      dataType: 'uint8',
      crop: { x: cropX, y: cropY, width: cropW, height: cropH },
    });
  };

  return {
    preprocessFrame,
    preprocessFrameForSeg,
    cropAndResize,
    cropAndResizeUint8,
    extractCroppedData,
    clamp,
    logWorklet,
  };
};

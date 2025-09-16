from PIL import Image
import numpy as np

# 1. RGB 이미지 열기
img = Image.open('apple.jpg').convert('RGB')

# 2. 모델 입력 크기 (MCUNet-in3 기준: 64x64)
img = img.resize((64, 64))

# 3. numpy 배열로 변환 및 정규화 (float32)
img_data = np.asarray(img, dtype=np.float32) / 255.0  # 정규화

# 4. NHWC → 1xHxWxC 형태로 배치 차원 추가 (옵션)
img_data = np.expand_dims(img_data, axis=0)  # shape: (1, 64, 64, 3)

# 5. raw 바이너리 저장 (float32 형식)
with open('apple_float32.raw', 'wb') as f:
    f.write(img_data.astype(np.float32).tobytes())

from PIL import Image
import numpy as np

# 1. RGB 이미지 열기
img = Image.open('banana_center.jpg').convert('RGB')

# 2. 모델 입력 크기로 조정
img = img.resize((224, 224))
img.save('banana_224x224.jpg')

# 3. numpy 배열로 변환 (uint8, [0, 255] 범위 유지)
img_data = np.asarray(img, dtype=np.uint8)

# 4. NHWC → 1xHxWxC 형태로 배치 차원 추가
img_data = np.expand_dims(img_data, axis=0)  # shape: (1, 224, 224, 3)

# 5. [0, 255] 범위를 [-128, 127] 범위로 매핑 후 int8로 변환하여 저장
# (uint8에서 바로 int8로 바꾸면 값이 깨지므로, 더 큰 타입으로 변환 후 연산)
int8_data = (img_data.astype(np.int16) - 128).astype(np.int8)

with open('banana_int8_224.raw', 'wb') as f:
    f.write(int8_data.tobytes())

print("Correctly converted and saved as int8 raw file.")
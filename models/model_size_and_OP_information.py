import os
import tensorflow as tf
from collections import Counter

tfl_path = "emo_mobilebert_int8.tflite"

# 파일 사이즈
size_bytes = os.path.getsize(tfl_path)
size_mb = size_bytes / (1024*1024)
print(f"[Model size] {size_bytes} bytes ({size_mb:.2f} MB)")

# OP 정보 수집
interpreter = tf.lite.Interpreter(model_path=tfl_path)
# allocate_tensors() 전에 호출해도 되지만, 일부 버전은 allocate 후에 더 안전
interpreter.allocate_tensors()

# 비공개 API지만, 가장 간단하게 OP 이름을 얻는 방법
ops = interpreter._get_ops_details()  # list of dicts: {'op_name': '...', 'inputs': [...], 'outputs': [...]}

op_names = [op['op_name'] for op in ops]
unique_ops = sorted(set(op_names))
op_counts = Counter(op_names)

print("\n[Unique OP types]")
for name in unique_ops:
    print(f" - {name}")

print("\n[OP counts]")
for name, cnt in op_counts.most_common():
    print(f" {name:30s} : {cnt}")


import numpy as np
from transformers import AutoTokenizer

text = "I feel really happy today!"
tokenizer = AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
SEQUENCE_LENGTH = 32

inputs = tokenizer(text, return_tensors="np", max_length=SEQUENCE_LENGTH, truncation=True, padding="max_length")
# 입력 id만 추출
input_ids = inputs["input_ids"].astype(np.int32)

# 바이너리 파일로 저장
input_ids.tofile("emo_mobilebert_input.raw")
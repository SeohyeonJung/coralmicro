import numpy as np
from transformers import AutoTokenizer

model_name = "csarron/mobilebert-uncased-squad-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

question = "Where is the Eiffel Tower?"
context  = "The Eiffel Tower is located in Paris, France."
max_len = 64  # C++에서 사용하는 kSequenceLength와 동일

inputs = tokenizer(
    question,
    context,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="np"
)

# INT8 모델은 input_ids, attention_mask, token_type_ids를 int8로 저장
input_ids      = inputs["input_ids"].astype(np.int8)
attention_mask = inputs["attention_mask"].astype(np.int8)
token_type_ids = inputs["token_type_ids"].astype(np.int8)

# C++에서 읽을 수 있도록 raw 파일로 저장
with open("mobilebert_qa_input.raw", "wb") as f:
    f.write(input_ids.tobytes())
    f.write(attention_mask.tobytes())
    f.write(token_type_ids.tobytes())

print("✅ Input .raw 파일 생성 완료")
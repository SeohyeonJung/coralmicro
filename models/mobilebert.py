import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

# =============================
# 1️⃣ 모델 및 토크나이저 로드
# =============================
model_name = "csarron/mobilebert-uncased-squad-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tf_model = TFAutoModelForQuestionAnswering.from_pretrained(model_name, from_pt=True)

SEQUENCE_LENGTH = 64
BATCH_SIZE = 1

# =============================
# 2️⃣ 예측 테스트 (기존 코드와 동일)
# =============================
question = "Where is the Eiffel Tower?"
context = "The Eiffel Tower is located in Paris, France."
inputs = tokenizer(question, context, return_tensors="tf", max_length=SEQUENCE_LENGTH, truncation=True)
outputs = tf_model(inputs)
start_index = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
end_index = tf.argmax(outputs.end_logits, axis=1).numpy()[0]
answer_tokens = inputs["input_ids"][0][start_index:end_index+1]
answer = tokenizer.decode(answer_tokens)
print("Answer (Test):", answer)

# =============================
# 3️⃣ Concrete Function 생성 (Static Shape)
# =============================
@tf.function(input_signature=[
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32, name="input_ids"),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32, name="attention_mask"),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32, name="token_type_ids")
])
def static_concrete_func(input_ids, attention_mask, token_type_ids):
    outputs = tf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )
    # start_logits, end_logits 그대로 반환
    return {
        "start_logits": outputs.start_logits,
        "end_logits": outputs.end_logits
    }

concrete_func = static_concrete_func.get_concrete_function()

# =============================
# 4️⃣ TFLite 변환기 생성
# =============================
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# INT8 양자화 설정
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# =============================
# 5️⃣ Representative Dataset 정의 (INT8 양자화용)
# =============================
vocab_size = tokenizer.vocab_size
def representative_dataset():
    for _ in range(100):
        yield [
            np.random.randint(0, vocab_size, size=(BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.int32),
            np.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.int32),
            np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.int32)
        ]

converter.representative_dataset = representative_dataset

# =============================
# 6️⃣ 변환 수행 및 저장
# =============================
TFLITE_OUT = "mobilebert_qa_squad_int8_static.tflite"

try:
    tflite_model = converter.convert()
    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite INT8 모델 변환 완료: '{TFLITE_OUT}'")
except Exception as e:
    print("❌ 변환 실패:", e)

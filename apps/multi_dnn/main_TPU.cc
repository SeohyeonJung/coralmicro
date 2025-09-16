// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/tasks.h"
#include "libs/base/timer.h"

#include "libs/tpu/edgetpu_manager.h"
#include "libs/tensorflow/posenet.h"  // Posenet custom decoder op
#include "libs/tensorflow/utils.h"

#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/kernels/micro_ops.h"

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace coralmicro {
namespace {

// arena size
constexpr int kTensorArenaSize1 = 2 * 1024 * 1024;
constexpr UBaseType_t kAppTaskPriority = tskIDLE_PRIORITY + 1;  // 추가
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena1, kTensorArenaSize1);

// Edge TPU-compiled posenet model + test input
constexpr char kModelPath1[]   = "/models/bert_tiny_relu_fp32_bs1_L128_edgetpu.tflite";
// constexpr char kInputRawPath1[] = "/models/posenet_test_input_324.bin";

/////////// TASK 1 ///////////
[[noreturn]] void TASK1(void* param) {
  vTaskDelay(pdMS_TO_TICKS(1000));

  // 1) Open Edge TPU
  printf("Step 1: Open TPU\r\n");
  auto tpu_ctx = EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
  if (!tpu_ctx) {
    printf("TASK1: Failed to open Edge TPU device\r\n");
    vTaskSuspend(nullptr);
  }

  // 2) Load model
  printf("Step 2: Load model\r\n");
  std::vector<uint8_t> model_buf;
  if (!LfsReadFile(kModelPath1, &model_buf)) {
    printf("TASK1: Failed to read model: %s\r\n", kModelPath1);
    vTaskSuspend(nullptr);
  }
  const tflite::Model* model = tflite::GetModel(model_buf.data());

  flatbuffers::Verifier v(model_buf.data(), model_buf.size());
  if (!tflite::VerifyModelBuffer(v)) {
    printf("TASK1: VerifyModelBuffer FAILED (not a valid TFLite flatbuffer)\r\n");
  }
  printf("TASK1: model->version() = %lu, expected = %d\r\n",
        model->version(), TFLITE_SCHEMA_VERSION);

  // 3) Resolver & Interpreter
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());  // Edge TPU op only
  resolver.AddTranspose();                         

  tflite::MicroInterpreter interpreter(model, resolver,
                                      tensor_arena1, kTensorArenaSize1,
                                      &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("TASK1: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  // 4) Load input
  /*
  TfLiteTensor* input = interpreter.input(0);
  if (!LfsReadFile(kInputRawPath1, tflite::GetTensorData<uint8_t>(input), input->bytes)) {
    printf("TASK2: Failed to load input: %s (expect %d bytes)\r\n",
           kInputRawPath1, input->bytes);
    vTaskSuspend(nullptr);
  }
  */
  const int num_inputs = interpreter.inputs_size();
  printf("TASK1: inputs_size = %d\r\n", num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    TfLiteTensor* t = interpreter.input(i);
    printf("  input[%d]: name=%s, bytes=%d, type=%d\r\n",
           i, t->name ? t->name : "(null)", t->bytes, t->type);
  }

  // 파일 경로 매핑 
  const char* kInputFiles[3] = {
      "/models/input_ids.raw",
      "/models/attention_mask.raw",
      "/models/token_type_ids.raw",
  };

  // 안전장치: 모델이 3개 입력이라고 가정하지만, 다르면 최소한 인덱스 범위 체크
  const int kExpectedInputs = 3;
  if (num_inputs < kExpectedInputs) {
    printf("TASK1: Model expects %d inputs, but got %d\r\n", kExpectedInputs, num_inputs);
    vTaskSuspend(nullptr);
  }

  // 각 입력에 파일 로딩
  for (int i = 0; i < kExpectedInputs; ++i) {
    TfLiteTensor* t = interpreter.input(i);

    // 입력이 int32일 확률이 높음 (토큰 인덱스/마스크)
    // raw는 바이트 스트림이므로 uint8_t*로 읽어도 메모리에 그대로 들어감.
    if (!LfsReadFile(kInputFiles[i], reinterpret_cast<uint8_t*>(t->data.raw), t->bytes)) {
      printf("TASK1: Failed to load input[%d] from %s (expect %d bytes)\r\n",
             i, kInputFiles[i], t->bytes);
      vTaskSuspend(nullptr);
    } else {
      printf("TASK1: Loaded input[%d] from %s (%d bytes)\r\n",
             i, kInputFiles[i], t->bytes);
    }
  }

  // 5) Inference loop + PERF
  std::vector<float> times_ms;
  //for (int n = 0; n < 30; ++n) {
  for (int n = 0; n < 5; ++n) {
    float t0 = static_cast<float>(TimerMicros());
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("TASK1: Inference failed at %d\r\n", n + 1);
    } else {
      float t1 = static_cast<float>(TimerMicros());
      float lat = (t1 - t0) / 1000.0f;
      times_ms.push_back(lat);
      float thr = 1000.0f / lat;
      printf("Task 1 %2dth latency: %.3f ms, throughput: %.2f inferences/sec\r\n",
             n + 1, lat, thr); 
    }
    taskYIELD();
  }

  // 6) Summary
  /*
  float sum = 0.f; for (float t : times_ms) sum += t;
  float avg = times_ms.empty() ? 0.f : sum / times_ms.size();
  float avg_thr = (avg > 0.f) ? 1000.f / avg : 0.f;
  printf("=== Task 2 Performance Summary (TPU/Classifier) ===\r\n");
  printf("Average Latency  : %.3f ms\r\n", avg);
  printf("Average Throughput: %.2f inferences/sec\r\n", avg_thr);
  */
  vTaskDelete(nullptr);
  
}

/////////// mian //////////////
[[noreturn]] void Main() {
  LedSet(Led::kUser, true);

  xTaskCreate(TASK1, "task1_tpu", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);

  vTaskSuspend(nullptr); 
}

}  // namespace 
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
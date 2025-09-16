// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/tasks.h"
#include "libs/base/timer.h"

#include "libs/tensorflow/utils.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/all_ops_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/kernels/micro_ops.h"

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "person_detect.h"
#include "mcu.h"

#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdint>

// Runs face detection on the Edge TPU, using the on-board camera, printing
// results to the serial console and turning on the User LED when a face
// is detected.
//
// 1) Change model, resolver to load, 
// 2) Change CMakeLists.txt model to load

namespace coralmicro {
namespace {


constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
constexpr UBaseType_t kAppTaskPriority = tskIDLE_PRIORITY + 1;  // 추가
constexpr int kTensorArenaSize1 = 1024 * 1024;
// constexpr int kTensorArenaSize2 = 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena1, kTensorArenaSize1);
// STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena2, kTensorArenaSize2);

constexpr char kModelPath[] =
    //"/models/person_detect_model.tflite";
    "/models/bert_tiny_tanh_manual_INT8_bs1_L128_outint8.tflite";

// constexpr char kModelPath2[] = "/models/mcunet-320kb-1mb_imagenet.tflite";


[[noreturn]] void TASK1(void* param) {
  vTaskDelay(pdMS_TO_TICKS(1000));

  /////////// Load Model ///////////
  std::vector<uint8_t> model;

  if (!LfsReadFile(kModelPath, &model)) {
    printf("TASK1: Failed to read model: %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  printf("TASK1: Model load done\r\n");
  fflush(stdout);
  //////////////////////////////////

  /////////// Interprete ///////////
  tflite::MicroErrorReporter error_reporter;
  
  // tinybert
  tflite::MicroMutableOpResolver<16> resolver;
  
  resolver.AddTanh(); 
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddCast();
  resolver.AddSub();
  resolver.AddMul();
  resolver.AddGather();
  resolver.AddFullyConnected();
  resolver.AddAdd();
  resolver.AddTranspose();
  resolver.AddMean();
  resolver.AddRsqrt();
  resolver.AddSquaredDifference();
  resolver.AddQuantize();
  // resolver.AddBatchMatmul();
  
  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena1, kTensorArenaSize1, &error_reporter);
  //////////////////////////////////

  //////// AllocateTensor /////////
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("TASK1: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  // input .raw
  const int num_inputs = interpreter.inputs_size();
  printf("inputs=%d\r\n", num_inputs);

  // 타입/바이트/shape 확인(디버그용)
  for (int i = 0; i < num_inputs; ++i) {
    const TfLiteTensor* t = interpreter.input_tensor(i);
    printf("  input[%d]: type=%d, bytes=%d, dims=[", i, (int)t->type, t->bytes);
    for (int d = 0; d < t->dims->size; ++d)
      printf("%d%s", t->dims->data[d], (d+1 < t->dims->size ? "," : ""));
    printf("]\r\n");
  }

  auto load_raw_exact = [&](const char* path, void* dst, size_t nbytes) -> bool {
    if (!LfsReadFile(path, reinterpret_cast<uint8_t*>(dst), nbytes)) {
      printf("Failed to load: %s (need %zu bytes)\r\n", path, nbytes);
      return false;
    }
    return true;
  };

  const char* kFiles[3] = {
    "/models/attention_mask.raw",   // input[0] = attention_mask
    "/models/input_ids.raw",        // input[1] = input_ids
    "/models/token_type_ids.raw"    // input[2] = token_type_ids
  };

  bool ok = true;
  if (num_inputs == 3) {
    for (int i = 0; i < 3; ++i) {
      TfLiteTensor* tin = interpreter.input_tensor(i);
      if (tin->type != kTfLiteInt32) {
        printf("WARN: input[%d] type=%d (expected INT32=3). bytes=%d\r\n",
              i, tin->type, tin->bytes);
      }
      ok &= load_raw_exact(kFiles[i], tflite::GetTensorData<void>(tin), tin->bytes);
    }
  } else if (num_inputs == 1) {

    TfLiteTensor* tin = interpreter.input_tensor(0);
    ok &= load_raw_exact("/models/mobilebert_emo_input.raw",
                        tflite::GetTensorData<void>(tin), tin->bytes);
  } else {
    printf("Unexpected inputs count: %d\r\n", num_inputs);
    ok = false;
  }

  if (!ok) {
    printf("Input load failed. Abort.\r\n");
    vTaskSuspend(nullptr);
  }

  /*
  auto* input = interpreter.input_tensor(0);
  if (!LfsReadFile("/models/mobilebert_emo_input.raw", tflite::GetTensorData<uint8_t>(input),
                    input->bytes)) {
    printf("TASK1: Failed to load input image\r\n");
    vTaskSuspend(nullptr);
  }
  */

  /////////// Inference ////////////
  std::vector<float> time;
  int n = 0;
  // while (n < 30) {
  while (n < 5) {
    float start = static_cast<float>(TimerMicros());
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("TASK1: Inference failed\r\n");
    } else {
      printf("Task 1 %dth latency: %.6f ms\r\n", n+1, (static_cast<float>(TimerMicros()) - start)/1000);
      // fflush(stdout);
    }
    // vTaskDelay(pdMS_TO_TICKS(1));
    taskYIELD();
    n++;
  }

  vTaskDelete(NULL);
  //////////////////////////////////
}

[[noreturn]] void Main() {
  // Task 시작 부분
  LedSet(Led::kUser, 1);  // User LED ON

  xTaskCreate(TASK1, "task1", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);
  // xTaskCreate(TASK2, "task2", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);
  vTaskSuspend(NULL);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}


/*
[[noreturn]] void TASK2(void* param) {
  vTaskDelay(pdMS_TO_TICKS(1000));

  /////////// Load Model ////////////
  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath2, &model)) {
    printf("TASK2: Failed to read model: %s\r\n", kModelPath2);
    vTaskSuspend(nullptr);
  }
  printf("TASK2: Model load done\r\n");
  fflush(stdout);
  //////////////////////////////////

  //////////// Interprete //////////
  tflite::MicroErrorReporter error_reporter;

  // mcunet
  tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddPad();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddAdd();
  resolver.AddAveragePool2D();
 
  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena2, kTensorArenaSize2, &error_reporter);

  ///////// AllocateTensor /////////
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("TASK2: AllocateTensors() failed\r\n");
        vTaskSuspend(nullptr);
  }
  
  TfLiteTensor* input = interpreter.input(0);


  // Input setting for MCUNET
  if (!LfsReadFile("/models/apple_float32.raw", tflite::GetTensorData<uint8_t>(input),
                    input->bytes)) {
    printf("TASK2: Failed to load input image\r\n");
    vTaskSuspend(nullptr);
  }
  //////////////////////////////////

  //////////// Inference ///////////
  std::vector<float> time;
  int n = 0;
  // while (true) {
  while (n < 30) {
    float start = static_cast<float>(TimerMicros());
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("TASK2: Inference failed\r\n");
    } else {
      // time.push_back((static_cast<float>(TimerMicros()) - start)/1000);
      printf("Task 2 %dth latency: %.6f ms\r\n", n+1, (static_cast<float>(TimerMicros()) - start)/1000);

    }
    // vTaskDelay(pdMS_TO_TICKS(1));
    taskYIELD();
    n++;
  }

  vTaskDelete(NULL);
  ////////////////////////////////////
}
*/
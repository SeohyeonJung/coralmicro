// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/tasks.h"
#include "libs/base/timer.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tensorflow/posenet.h"  // Posenet custom decoder op

#include "libs/tensorflow/utils.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/all_ops_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#include <cstdio>
#include <vector>
#include <cstring>
#include <cstdint>

namespace coralmicro {
namespace {


// 이 줄을 추가하여 태스크 우선순위를 정의합니다.
constexpr int kAppTaskPriority = tskIDLE_PRIORITY + 1;

constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
constexpr int kTensorArenaSize1 = 1024 * 1024;
// constexpr int kTensorArenaSize2 = 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena1, kTensorArenaSize1);
// STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena2, kTensorArenaSize2);

constexpr char kModelPath[] =
    // "/models/emo_mobilebert_int8_edgetpu.tflite";
    "/models/bert_tiny_relu_fp32_bs1_L128_edgetpu.tflite";

constexpr char kInputRawPath[] = "/models/mobilebert_qa_input.raw";
// constexpr char kModelPath2[] =
    // "/models/mcunet-320kb-1mb_imagenet.tflite";
    // "/models/trained_lstm_int8.tflite";

[[noreturn]] void TASK1(void* param) {
    const int num_inferences = 30;
    const int BANANA_INDEX = 955;
    std::vector<int> accuracy_table;
    std::vector<float> times_ms;
    accuracy_table.reserve(num_inferences);
    times_ms.reserve(num_inferences);

    vTaskDelay(pdMS_TO_TICKS(1000));

    // Edge tpu context 
    auto tpu_context =
        EdgeTpuManager::GetSingleton()->OpenDevice(PerformanceMode::kMax);
    if (!tpu_context) {
        printf("Failed to get tpu context.\r\n");
        vTaskSuspend(nullptr);
    }

    /////////// Load Model ///////////
    std::vector<uint8_t> model; // store for flatbuffer model
    if (!LfsReadFile(kModelPath, &model)) {
        printf("TASK1: Failed to read model: %s\r\n", kModelPath);
        vTaskSuspend(nullptr);
    }

    printf("TASK1: Model load done\r\n");
    fflush(stdout);
    //////////////////////////////////

    const tflite::Model* model_data = tflite::GetModel(model.data());
    flatbuffers::Verifier v(model.data(), model.size());
    if (!tflite::VerifyModelBuffer(v)) {
        printf("TASK1: VerifyModelBuffer FAILED (not a valid TFLite flatbuffer)\r\n");
    }
    printf("TASK1: model->version() = %lu, expected = %d\r\n",
        model_data->version(), TFLITE_SCHEMA_VERSION);

    /////////// Interprete ///////////
    tflite::MicroErrorReporter error_reporter;

    // efficientnet-edgetpu-S_quant
    tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());
    resolver.AddReshape();    
    resolver.AddCast();       
    resolver.AddQuantize();   
    resolver.AddGather();   

    tflite::MicroInterpreter interpreter(model_data, resolver,
                                        tensor_arena1, kTensorArenaSize1, &error_reporter);

    //////// AllocateTensor /////////
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("TASK1: AllocateTensors() failed\r\n");
        vTaskSuspend(nullptr);
    }

    size_t used_bytes = interpreter.arena_used_bytes();
    printf("Tensor Arena Usage: %u bytes (%.2f KB)\r\n", 
           used_bytes, static_cast<float>(used_bytes) / 1024.0f);
    fflush(stdout);


    TfLiteTensor* input = interpreter.input(0);

    if (!LfsReadFile(kInputRawPath, tflite::GetTensorData<uint8_t>(input),
                    input->bytes)) {
        printf("TASK1: Failed to load input image\r\n");
        vTaskSuspend(nullptr);
    }
    //////////////////////////////////

    /////////// Inference ////////////
    printf("TASK1: Model invoke start\r\n");
    fflush(stdout);

    for (int n = 0; n < num_inferences; ++n) {
        float start = static_cast<float>(TimerMicros());
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("TASK1: Inference failed\r\n");
        } else {
            printf("TASK1: Model %d invoke done\r\n", n+1);
            fflush(stdout);
            // auto* output = interpreter.output(0);
            auto* output = interpreter.output(0);
            int8_t* output_data = output->data.int8;
            // 출력 텐서의 크기 (클래스 개수, ImageNet의 경우 1000)
            const int output_size = output->bytes;

            // 3. 가장 높은 점수를 가진 클래스 찾기
            int top_class_index = 0;
            int8_t max_score = -128;
            for (int i = 0; i < output->bytes; ++i) {
                if (output_data[i] > max_score) {
                    max_score = output_data[i];
                    top_class_index = i;
                }
            }
            accuracy_table.push_back(top_class_index);

            // 4. 최종 결과 출력
            float latency_ms = (static_cast<float>(TimerMicros()) - start) / 1000.0f;
            // printf("Task 1 %dth latency: %.6f ms\r\n", n+1, (static_cast<float>(TimerMicros()) - start)/1000);
            times_ms.push_back(latency_ms);

            fflush(stdout);
        }
        taskYIELD();
        // vTaskDelay(pdMS_TO_TICKS(1));
    }
    printf("\n=========== Inference Completed ===========\r\n");
    int correct_predictions = 0;
    int i=0;
    for (int result_index : accuracy_table) {
        printf("%d's result_index: %d\n", i++, result_index);
        if (result_index == BANANA_INDEX) {
            correct_predictions++;
        }
    }

    float accuracy_percent = (static_cast<float>(correct_predictions) / num_inferences) * 100.0f;
    printf("Accuracy for 'banana' (Index %d):\r\n", BANANA_INDEX);
    printf("Correct: %d / %d (%.2f%%)\r\n", correct_predictions, num_inferences, accuracy_percent);
  
    float sum = 0.f; for (float t : times_ms) sum += t;
    float avg = times_ms.empty() ? 0.f : sum / times_ms.size();
    float avg_thr = (avg > 0.f) ? 1000.f / avg : 0.f;
    printf("=== Task 1 Performance Summary (TPU/Posenet) ===\r\n");
    printf("Average Latency   : %.3f ms\r\n", avg);
    printf("Average Throughput: %.2f inferences/sec\r\n", avg_thr);
    printf("============================================\r\n");
    fflush(stdout);

    vTaskDelete(NULL);
    //////////////////////////////////
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
  
  // LSTM
  // tflite::MicroMutableOpResolver<4> resolver;
  // resolver.AddUnidirectionalSequenceLSTM();
  // resolver.AddReshape();
  // resolver.AddFullyConnected();
  // resolver.AddSoftmax();

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

  // tflite::MicroInterpreter interpreter(tflite::GetModel(mcunet_320kb_1mb_imagenet_tflite), resolver,
  //                                      tensor_arena2, kTensorArenaSize2, &error_reporter);

  //////////////////////////////////

  ///////// AllocateTensor /////////
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("TASK2: AllocateTensors() failed\r\n");
        vTaskSuspend(nullptr);
  }
  
  TfLiteTensor* input = interpreter.input(0);

  // Input setting for LSTM
  // std::vector<uint8_t> input_data(input->bytes, 0x1);  // 예시로 중간값(64)으로 채움
  // memcpy(input->data.uint8, input_data.data(), input->bytes);

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
      // fflush(stdout);
      // TfLiteTensor* output = interpreter.output(0);

      // printf("%d's [latency, result]: [%.6fms, %d]\r\n", n++, elapsed_ms, output->data.uint8[0]);
      // int8_t* output_data = output->data.int8;
      // int output_size = output->bytes; // 보통 1000 바이트, 즉 1000개 클래스

      // 예) 가장 높은 점수(최댓값) 찾기
      // int max_index = 0;
      // int8_t max_value = output_data[0];
      // for (int i = 1; i < output_size; i++) {
      //     if (output_data[i] > max_value) {
      //         max_value = output_data[i];
      //         max_index = i;
      //     }
      // }
    }
    // vTaskDelay(pdMS_TO_TICKS(1));
    taskYIELD();
    n++;
  }
  // for(unsigned int i=0; i < time.size(); i++){
  //   printf("Task 2 %dth latency: %.6f ms\r\n", i+1, time[i]);
  //   fflush(stdout);
  // }
  vTaskDelete(NULL);
  ////////////////////////////////////
}
*/

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

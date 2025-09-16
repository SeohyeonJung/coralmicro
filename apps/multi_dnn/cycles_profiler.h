// cycles_profiler.h
#pragma once

#include <stdint.h>
#include <cstddef>
#include <cstdio>

#include "profiling_utils.h" 
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_profiler.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"

// DWT CYCCNT 기반의 간단 MicroProfiler 구현
class CyclesProfiler : public tflite::MicroProfiler {
 public:
  explicit CyclesProfiler(tflite::ErrorReporter* er, uint32_t cpu_hz)
      : er_(er), cpu_hz_(cpu_hz) {}

  // MicroProfiler의 시그니처에 맞게 EventType 제거
  uint32_t BeginEvent(const char* tag) override {
    if (!enabled_) return 0;

    const uint32_t handle = next_id_++;
    const uint32_t slot   = handle % kMax;

    ids_[slot]    = handle;
    names_[slot]  = tag ? tag : "(op)";
    starts_[slot] = GetCycleCount();

    return handle;
  }

  void EndEvent(uint32_t handle) override {
    if (!enabled_) return;

    const uint32_t slot = handle % kMax;
    if (ids_[slot] != handle) {
      // 버퍼 랩어라운드 등으로 매칭 실패 시 무시
      return;
    }

    const uint32_t end   = GetCycleCount();
    const uint32_t start = starts_[slot];
    const uint32_t cyc   = end - start;

    const float ms = static_cast<float>(cyc) * 1000.0f / static_cast<float>(cpu_hz_);
    // TF_LITE_REPORT_ERROR(er_, "%s : %lu cycles (%.3f ms)", names_[slot], static_cast<unsigned long>(cyc), ms);
    printf("%s : %lu cycles (%.3f ms)\r\n", names_[slot], (unsigned long)cyc, ms);
  }

  void SetEnabled(bool on) { enabled_ = on; }

  void Reset() {
    next_id_ = 1;
    for (size_t i = 0; i < kMax; ++i) {
      ids_[i] = 0;
      names_[i] = nullptr;
      starts_[i] = 0;
    }
  }

 private:
    // 모델 op 수보다 조금 여유 있게. 필요시 1024 등으로 증가
    static constexpr uint32_t kMax = 512;

    tflite::ErrorReporter* er_;
    uint32_t cpu_hz_;
    bool     enabled_ = true;

    uint32_t next_id_ = 1;
    uint32_t ids_[kMax] = {};
    const char* names_[kMax] = {};
    uint32_t starts_[kMax] = {};
};

// profiling_utils.h
#pragma once
#include <stdint.h>

// 가능하면 보드 헤더를 먼저 끼워 CMSIS 심볼(CoreDebug/DWT 등)을 살립니다.
#if __has_include("mcu.h")
  #include "mcu.h"
#endif

// DWT CYCCNT 사용 활성화
static inline void EnableCycleCounter(void) {
#if defined(CoreDebug) && defined(DWT) && defined(CoreDebug_DEMCR_TRCENA_Msk) && defined(DWT_CTRL_CYCCNTENA_Msk)
  // CMSIS 경로 (권장)
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // Trace enable
  DWT->CYCCNT = 0;                                 // Reset
  DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;            // Counter enable
#else
  // 폴백: 레지스터 주소 직접 접근 (이름 충돌 피하려고 매크로 NO!)
  volatile uint32_t* DEMCR_REG      = (uint32_t*)0xE000EDFC;
  volatile uint32_t* DWT_CTRL_REG   = (uint32_t*)0xE0001000;
  volatile uint32_t* DWT_CYCCNT_REG = (uint32_t*)0xE0001004;
  *DEMCR_REG      |= (1u << 24);   // TRCENA
  *DWT_CYCCNT_REG  = 0;
  *DWT_CTRL_REG   |= (1u << 0);    // CYCCNTENA
#endif
}

static inline uint32_t GetCycleCount(void) {
#if defined(DWT)
  return DWT->CYCCNT;
#else
  volatile uint32_t* DWT_CYCCNT_REG = (uint32_t*)0xE0001004;
  return *DWT_CYCCNT_REG;
#endif
}

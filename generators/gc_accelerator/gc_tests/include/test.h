#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "encoding.h"
#include "macroInstHelper.h"
#include "marchid.h"

#define SHADOW_BASE 0xfec0000000UL
uint8_t *data = (uint8_t*)SHADOW_BASE;

static inline void *shadow_addr(uint64_t addr) {
    return (void *)addr;
}

static inline uint64_t load64(uintptr_t addr) {
    return *(volatile uint64_t *)shadow_addr(addr);
}

static inline void store64(uintptr_t addr, uint64_t val) {
	addr = (uintptr_t)shadow_addr(addr);
    *(volatile uint64_t *)addr = val;
}

static inline uint64_t load32(uintptr_t addr) {
    return *(volatile uint32_t *)shadow_addr(addr);
}

static inline void store32(uintptr_t addr, uint32_t val) {
	addr = (uintptr_t)shadow_addr(addr);
    *(volatile uint32_t *)addr = val;
}

static inline uint64_t load8(uintptr_t addr) {
    return *(volatile uint8_t *)shadow_addr(addr);
}

static inline void store8(uintptr_t addr, uint8_t val) {
	addr = (uintptr_t)shadow_addr(addr);
    *(volatile uint8_t *)addr = val;
}

struct HWGCParameters
{
  uint32_t chunkSize;
  uint32_t ageThreshold;
  uint32_t heapRegionBias;
  uint32_t regionAttrShiftBy;
  uint32_t heapRegionShiftBy;
  uint32_t logOfHRGrainBytes;
  uint64_t stepperOffset;
  uint64_t youngWordsBase;
  uint64_t regionAttrBase;
  uint64_t plabAllocatorPtr;
  uint64_t regionAttrBiasedBase;
  uint64_t heapRegionBiasedBase;
  uint64_t parScanThreadStatePtr;
  uint32_t taskQueueBottom;
  uint64_t taskQueueElemsBase;
  uint64_t humogousReclaimCandidateBoolBase;
  uint64_t cardTablePtr;
  uint64_t g1h;
  uint64_t intArrayKlass;
  uint64_t objectKlass;
  uint64_t lockPtr;
  uint64_t thread;
  uint64_t dummyRegion;
  uint64_t numaPtr;
  uint64_t compressedOopBase;
  uint64_t compressedKlassPointerBase;
  uint8_t compressedOopShift;
  uint8_t compressedKlassPointerShift;
  uint8_t useCompressedOops;
  uint8_t useCompressedKlassPointers;
};
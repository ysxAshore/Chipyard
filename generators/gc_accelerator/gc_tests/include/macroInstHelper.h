#include "rocc_interface.h"

// opcode == 0x5B(CUSTOM2)
// func == 0 write RegionAttrBase and RegionAttrBiasedBase
// func == 1 write RegionAttrShiftBy and HeapRegionBias and HeapRegionShiftBy
// func == 2 write HeapRegionBiasedBase and HumongousReclaimCandiatesBoolBase
// func == 3 write ParScanThreadStatePtr and TaskQueue_BottomAddr
// func == 4 write TaskQueue_AgeTopAddr and TaskQueue_ElemsBase
#define CONFIG_FUNC 0
#define CONFIG0_FUNC (CONFIG_FUNC + 0)
#define CONFIG1_FUNC (CONFIG_FUNC + 1)
#define CONFIG2_FUNC (CONFIG_FUNC + 2)
#define CONFIG3_FUNC (CONFIG_FUNC + 3)
#define CONFIG4_FUNC (CONFIG_FUNC + 4)
#define CONFIG5_FUNC (CONFIG_FUNC + 5)
#define CONFIG6_FUNC (CONFIG_FUNC + 6)
#define CONFIG7_FUNC (CONFIG_FUNC + 7)
#define CONFIG8_FUNC (CONFIG_FUNC + 8)
#define CONFIG9_FUNC (CONFIG_FUNC + 9)
#define CONFIG10_FUNC (CONFIG_FUNC + 10)
#define CONFIG11_FUNC (CONFIG_FUNC + 11)

// func == 16 inquire whether done?
#define INQUIRE_FUNC 16
#define INQUIRE_BUSY_FUNC (INQUIRE_FUNC + 0)
#define INQUIRE_RUN_CYCLES_FUNC (INQUIRE_FUNC + 1)
#define INQUIRE_READREQ_NUM_FUNC (INQUIRE_FUNC + 2)
#define INQUIRE_WRITEREQ_NUM_FUNC (INQUIRE_FUNC + 3)

void issue_config0(uint32_t chunkSize, uint32_t ageThreshold, uint32_t heapRegionBias, uint32_t regionAttrShiftBy){
    uint64_t rs1 = ((uint64_t)ageThreshold << 32) | chunkSize;
    uint64_t rs2 = ((uint64_t)regionAttrShiftBy << 32) | heapRegionBias;
    INS_XRR(rs1, rs2, CONFIG0_FUNC);
}

void issue_config1(uint32_t heapRegionShiftBy, uint32_t logOfHRGrainBytes, uint64_t stepperOffset){
    uint64_t rs1 = ((uint64_t)logOfHRGrainBytes << 32) | heapRegionShiftBy;
    INS_XRR(rs1, stepperOffset, CONFIG1_FUNC);
}

void issue_config2(uint64_t YoungWordsBase, uint64_t RegionAttrBase){
    INS_XRR(YoungWordsBase, RegionAttrBase, CONFIG2_FUNC);
}

void issue_config3(uint64_t PlabAllocatorPtr, uint64_t RegionAttrBiasedBase){
    INS_XRR(PlabAllocatorPtr, RegionAttrBiasedBase, CONFIG3_FUNC);
}

void issue_config4(uint64_t HeapRegionBiasedBase, uint64_t ParScanThreadStatePtr){
    INS_XRR(HeapRegionBiasedBase, ParScanThreadStatePtr, CONFIG4_FUNC);
}

void issue_config5(uint32_t TaskQueue_Bottom, uint64_t TaskQueue_ElemsBase){
    INS_XRR(TaskQueue_Bottom, TaskQueue_ElemsBase, CONFIG5_FUNC);
}

void issue_config6(uint64_t HumongousReclaimCandiatesBoolBase, uint64_t CardTablePtr){
    INS_XRR(HumongousReclaimCandiatesBoolBase, CardTablePtr, CONFIG6_FUNC);
}

void issue_config7(uint64_t G1h, uint64_t IntArrayKlassObj){
    INS_XRR(G1h, IntArrayKlassObj, CONFIG7_FUNC);
}

void issue_config8(uint64_t ObjectKlass, uint64_t LockPtr){
    INS_XRR(ObjectKlass, LockPtr, CONFIG8_FUNC);
}

void issue_config9(uint64_t Thread, uint64_t DummyRegion){
    INS_XRR(Thread, DummyRegion, CONFIG9_FUNC);
}

void issue_config10(uint64_t NumaPtr, uint64_t CompressedOopBase){
    INS_XRR(NumaPtr, CompressedOopBase, CONFIG10_FUNC);
}

void issue_config11(uint64_t CompressedKlassPointerBase, uint32_t CompressedFlag){
    INS_XRR(CompressedKlassPointerBase, CompressedFlag, CONFIG11_FUNC);
}

void issue_config(
    // config0
    uint32_t chunkSize,
    uint32_t ageThreshold,
    uint32_t heapRegionBias,
    uint32_t regionAttrShiftBy,
    // config1
    uint32_t heapRegionShiftBy,
    uint32_t logOfHRGrainBytes,
    uint64_t stepperOffset,
    // config2
    uint64_t YoungWordsBase,
    uint64_t RegionAttrBase,
    // config3
    uint64_t PlabAllocatorPtr,
    uint64_t RegionAttrBiasedBase,
    // config4
    uint64_t HeapRegionBiasedBase,
    uint64_t ParScanThreadStatePtr,
    // config5
    uint64_t TaskQueue_Bottom,
    uint64_t TaskQueue_ElemsBase,
    // config6
    uint64_t HumongousReclaimCandiatesBoolBase,
    uint64_t CardTablePtr,
    // config7
    uint64_t G1h,
    uint64_t IntArrayKlassObj,
    // config8
    uint64_t ObjectKlassPtr,   // 修正了原函数的语法错误
    uint64_t LockPtr,
    // config9
    uint64_t Thread,
    uint64_t DummyRegion,
    // config10
    uint64_t NumaPtr,
    uint64_t CompressedOopBase,
    // config11
    uint64_t CompressedKlassPointerBase,
    uint32_t CompressedFlag
) {
    issue_config0(chunkSize, ageThreshold, heapRegionBias, regionAttrShiftBy);
    issue_config1(heapRegionShiftBy, logOfHRGrainBytes, stepperOffset);
    issue_config2(YoungWordsBase, RegionAttrBase);
    issue_config3(PlabAllocatorPtr, RegionAttrBiasedBase);
    issue_config4(HeapRegionBiasedBase, ParScanThreadStatePtr);
    issue_config5(TaskQueue_Bottom, TaskQueue_ElemsBase);
    issue_config6(HumongousReclaimCandiatesBoolBase, CardTablePtr);
    issue_config7(G1h, IntArrayKlassObj);
    issue_config8(ObjectKlassPtr, LockPtr);
    issue_config9(Thread, DummyRegion);
    issue_config10(NumaPtr, CompressedOopBase);
    issue_config11(CompressedKlassPointerBase, CompressedFlag);
}

bool issue_inquireBusy(){
    uint64_t res = 1;
    INS_RRR(res, 0, 0, INQUIRE_BUSY_FUNC);
    return res;
}

uint64_t issue_inquireRunCycles(){
    uint64_t res = 0;
    INS_RRR(res, 0, 0, INQUIRE_RUN_CYCLES_FUNC);
    return res;
}

uint64_t issue_inquireReadReqNums(){
    uint64_t res = 0;
    INS_RRR(res, 0, 0, INQUIRE_READREQ_NUM_FUNC);
    return res;
}

uint64_t issue_inquireWriteReqNums(){
    uint64_t res = 0;
    INS_RRR(res, 0, 0, INQUIRE_WRITEREQ_NUM_FUNC);
    return res;
}

uint64_t readCycles(){
    uint64_t res = 0;
    asm volatile("rdcycle %0" :"=r"(res));
    return res;
}
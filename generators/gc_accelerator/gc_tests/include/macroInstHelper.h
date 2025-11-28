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

// func == 16 inquire whether done?
#define INQUIRE_FUNC 16
#define INQUIRE_BUSY_FUNC (INQUIRE_FUNC + 0)
#define INQUIRE_RUN_CYCLES_FUNC (INQUIRE_FUNC + 1)
#define INQUIRE_READREQ_NUM_FUNC (INQUIRE_FUNC + 2)
#define INQUIRE_WRITEREQ_NUM_FUNC (INQUIRE_FUNC + 3)

void issue_config0(uint64_t RegionAttrBase, uint64_t RegionAttrBiasedBase){
    INS_XRR(RegionAttrBase, RegionAttrBiasedBase, CONFIG0_FUNC);
}

void issue_config1(uint32_t RegionAttrShiftBy, uint32_t HeapRegionBias, uint32_t HeapRegionShiftBy){
    uint64_t rs1 = ((uint64_t)RegionAttrShiftBy << 32) | HeapRegionBias;
    INS_XRR(rs1, HeapRegionShiftBy, CONFIG1_FUNC);
}

void issue_config2(uint64_t HeapRegionBiasedBase, uint64_t HumongousReclaimCandiatesBoolBase){
    INS_XRR(HeapRegionBiasedBase, HumongousReclaimCandiatesBoolBase, CONFIG2_FUNC);
}

void issue_config3(uint64_t ParScanThreadStatePtr, uint64_t TaskQueue_BottomAddr){
    INS_XRR(ParScanThreadStatePtr, TaskQueue_BottomAddr, CONFIG3_FUNC);
}

void issue_config4(uint64_t TaskQueue_AgeTopAddr, uint64_t TaskQueue_ElemsBase){
    INS_XRR(TaskQueue_AgeTopAddr, TaskQueue_ElemsBase);
}

void issue_config(uint64_t RegionAttrBase, uint64_t RegionAttrBiasedBase, uint32_t RegionAttrShiftBy, uint32_t HeapRegionBias, uint32_t HeapRegionShiftBy, uint64_t HeapRegionBiasedBase, uint64_t HumongousReclaimCandiatesBoolBase, uint64_t ParScanThreadStatePtr, uint64_t TaskQueue_BottomAddr, uint64_t TaskQueue_AgeTopAddr, uint64_t TaskQueue_ElemsBase){
    issue_config0(RegionAttrBase, RegionAttrBiasedBase);
    issue_config1(RegionAttrShiftBy, HeapRegionBias, HeapRegionShiftBy);
    issue_config2(HeapRegionBiasedBase, HumongousReclaimCandiatesBoolBase);
    issue_config3(ParScanThreadStatePtr, TaskQueue_BottomAddr);
    issue_config4(TaskQueue_AgeTopAddr, TaskQueue_ElemsBase);
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


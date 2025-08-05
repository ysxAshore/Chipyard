package chipyard

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}
import freechips.rocketchip.subsystem.InTile
import cute._
// ---------------------
// BOOM V3 Configs
// Performant, stable baseline
// ---------------------

class CUTEShuttle512D512V512M256S1CoreConfig extends Config(
//   new cute.WithCUTE(Seq(0,1,2,3)) ++
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.CUTE_2Tops) ++
  new cute.WithCUTE(Seq(0)) ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(64) ++
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=40) ++
  new saturn.shuttle.WithShuttleVectorUnit(vLen = 512, dLen = 512, VectorParams.refParams, mLen = Option(512)) ++
  new shuttle.common.WithTCM(address = 0x70000000L, size = 2L << 20, banks = 2) ++
  // new cutev3.shuttle.WithShuttleTensorUnit(vLen = 512, dLen = 512, VectorParams.refParams, mLen = Option(512)) ++
  // new cutev3.common.WithCUTETCM ++
  // new cutev3.common.WithCUTESGTCM ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleDebugROB ++ // enable debug ROB
  new shuttle.common.WithShuttleDebugPrintf ++ // enable debug printf
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new chipyard.config.AbstractConfig)

class VerifyL2DramPerformenceTest1CUTEM256Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams.copy(Debug = CuteDebugParams.CMLDebugEnable)) ++
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 256) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)

class L2DramPerformenceTest1CUTEM64Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams) ++
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 64) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)

class L2DramPerformenceTest1CUTEM128Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams) ++
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 128) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)

class L2DramPerformenceTest1CUTEM256Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams) ++
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 256) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)

class L2DramPerformenceTest4CUTEM64Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams) ++
  new cute.WithCUTE(Seq(0,1,2,3)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(4) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 64) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)
class L2DramPerformenceTest4CUTEM128Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams) ++
  new cute.WithCUTE(Seq(0,1,2,3)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(4) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 128) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)

class L2DramPerformenceTest4CUTEM256Config extends Config(
  new cute.WithCuteCoustomParams(CoustomCuteParam = CuteParams.dram_L2_8Tops_PerformanceTestParams) ++
  new cute.WithCUTE(Seq(0,1,2,3)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(4) ++                          // small boom config
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(dataBits = 256) ++ //设置访存总线的位宽
  new freechips.rocketchip.subsystem.WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=100) ++
  new chipyard.config.AbstractConfig)

class YJPFPGACUTESmallBoomConfig extends Config(
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBoomsMMIOSpeedUp(1) ++                          // small boom config
  new chipyard.config.YJPAbstractConfig)

class CUTESmallBoomConfig extends Config(
  new cute.WithCUTE(Seq(0)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new chipyard.config.AbstractConfig)

class CUTEShuttle512D512V256M4CoreConfig extends Config(
  new cute.WithCUTE(Seq(0,1,2,3)) ++
  new saturn.shuttle.WithShuttleVectorUnit(512, 512, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithTCM ++
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(4) ++
  new chipyard.config.AbstractConfig)

class CUTEShuttle512D512V256M2Shuttle1BoomConfig extends Config(
  new cute.WithCUTE(Seq(1,2)) ++
  new saturn.shuttle.WithShuttleVectorUnit(512, 512, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithTCM ++
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
//   new shuttle.common.WithShuttleDebugROB ++ // enable debug ROB
//   new shuttle.common.WithShuttleDebugPrintf ++ // enable debug printf
  new shuttle.common.WithNShuttleCores(2) ++
//   new boom.v3.common.WithBoomCommitLogPrintf ++ // enable commit log printf
  // new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
  new chipyard.config.AbstractConfig)

class CUTETestConfig128bitdram512bitL2Widen3issueBoom extends Config(
// new freechips.rocketchip.subsystem.WithNBanks(8) ++
//   new freechips.rocketchip.subsystem.WithInclusiveCache(nWays=16, capacityKB=2048) ++
//   new freechips.rocketchip.subsystem.WithNMemoryChannels(4) ++
  new boom.bobcat.common.WithVector(3) ++
  new cute.WithCUTE(Seq(0)) ++
  // new freechips.rocketchip.subsystem.WithNBanks(2) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=40) ++
  new freechips.rocketchip.subsystem.WithNMemoryChannels(1) ++
//   new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=4096) ++
  new boom.bobcat.common.WithBoomCommitLogPrintf ++
  new boom.bobcat.common.WithBoomMemtraceLogPrintf ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new boom.bobcat.common.WithN3IssueWidenBooms(1) ++                    // 3 issue boom config
  new chipyard.config.AbstractConfig)


class CUTEShuttle512D512V256M1CoreConfig extends Config(
//   new cute.WithCUTE(Seq(0,1,2,3)) ++
  new saturn.shuttle.WithShuttleVectorUnit(512, 512, VectorParams.refParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithTCM ++
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class CUTEShuttle1024D1024V512M1CoreConfig extends Config(
//   new cute.WithCUTE(Seq(0,1,2,3)) ++
  new saturn.shuttle.WithShuttleVectorUnit(vLen = 1024, dLen = 1024, VectorParams.refParams, mLen = Option(512)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithTCM ++
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

class CUTEShuttle1024D1024V1024M256S1CoreConfig extends Config(
//   new cute.WithCUTE(Seq(0,1,2,3)) ++
  new saturn.shuttle.WithShuttleVectorUnit(vLen = 1024, dLen = 1024, VectorParams.refParams, mLen = Option(1024)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  // new shuttle.common.WithTCM ++
  // new shuttle.common.WithSGTCM ++
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new chipyard.config.AbstractConfig)

// class SmallBoomV3Config extends Config(
//   new boom.v3.common.WithNSmallBooms(1) ++                          // small boom config
//   new chipyard.config.AbstractConfig)

// class MediumBoomV3Config extends Config(
//   new boom.v3.common.WithNMediumBooms(1) ++                         // medium boom config
//   new chipyard.config.AbstractConfig)

// class LargeBoomV3Config extends Config(
//   new boom.v3.common.WithNLargeBooms(1) ++                          // large boom config
//   new chipyard.config.WithSystemBusWidth(128) ++
//   new chipyard.config.AbstractConfig)

// class MegaBoomV3Config extends Config(
//   new boom.v3.common.WithNMegaBooms(1) ++                           // mega boom config
//   new chipyard.config.WithSystemBusWidth(128) ++
//   new chipyard.config.AbstractConfig)

// class DualSmallBoomV3Config extends Config(
//   new boom.v3.common.WithNSmallBooms(2) ++                          // 2 boom cores
//   new chipyard.config.AbstractConfig)

// class Cloned64MegaBoomV3Config extends Config(
//   new boom.v3.common.WithCloneBoomTiles(63, 0) ++
//   new boom.v3.common.WithNMegaBooms(1) ++                           // mega boom config
//   new chipyard.config.WithSystemBusWidth(128) ++
//   new chipyard.config.AbstractConfig)

// class LoopbackNICLargeBoomV3Config extends Config(
//   new chipyard.harness.WithLoopbackNIC ++                        // drive NIC IOs with loopback
//   new icenet.WithIceNIC ++                                       // build a NIC
//   new boom.v3.common.WithNLargeBooms(1) ++
//   new chipyard.config.WithSystemBusWidth(128) ++
//   new chipyard.config.AbstractConfig)

// class MediumBoomV3CosimConfig extends Config(
//   new chipyard.harness.WithCospike ++                            // attach spike-cosim
//   new chipyard.config.WithTraceIO ++                             // enable the traceio
//   new boom.v3.common.WithNMediumBooms(1) ++
//   new chipyard.config.AbstractConfig)

// class dmiCheckpointingMediumBoomV3Config extends Config(
//   new chipyard.config.WithNPMPs(0) ++                            // remove PMPs (reduce non-core arch state)
//   new chipyard.harness.WithSerialTLTiedOff ++                    // don't attach anything to serial-tl
//   new chipyard.config.WithDMIDTM ++                              // have debug module expose a clocked DMI port
//   new boom.v3.common.WithNMediumBooms(1) ++
//   new chipyard.config.AbstractConfig)

// class dmiMediumBoomV3CosimConfig extends Config(
//   new chipyard.harness.WithCospike ++                            // attach spike-cosim
//   new chipyard.config.WithTraceIO ++                             // enable the traceio
//   new chipyard.harness.WithSerialTLTiedOff ++                    // don't attach anythint to serial-tl
//   new chipyard.config.WithDMIDTM ++                              // have debug module expose a clocked DMI port
//   new boom.v3.common.WithNMediumBooms(1) ++
//   new chipyard.config.AbstractConfig)

// class SimBlockDeviceMegaBoomV3Config extends Config(
//   new chipyard.harness.WithSimBlockDevice ++                     // drive block-device IOs with SimBlockDevice
//   new testchipip.iceblk.WithBlockDevice ++                       // add block-device module to peripherybus
//   new boom.v3.common.WithNMegaBooms(1) ++                        // mega boom config
//   new chipyard.config.WithSystemBusWidth(128) ++
//   new chipyard.config.AbstractConfig)

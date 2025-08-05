package chipyard

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}
import freechips.rocketchip.subsystem.InTile
import freechips.rocketchip.subsystem.WithNBanks
import cute._
import spire.std.seq
// ---------------------
// BOOM V3 Configs
// Performant, stable baseline
// ---------------------

class CUTEv3Shuttle512D512V512M256S4CoreConfig extends Config(
//   new cute.WithCUTE(Seq(0,1,2,3)) ++
  new cute.WithCuteCoustomParams(CoustomCuteParam = Cutev3Params.CUTE_8Tops_128SCP) ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(256) ++
  new WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=40) ++
  new cutev3.shuttle.WithShuttleTensorUnit(vLen = 512, dLen = 512, VectorParams.refParams, mLen = Option(512),cores = Seq(0,1,2,3)) ++
  new cutev3.common.WithCUTETCM ++
  new cutev3.common.WithCUTESGTCM ++
  new chipyard.config.WithSystemBusWidth(256) ++
  // new shuttle.common.WithShuttleDebugROB ++ // enable debug ROB
  // new shuttle.common.WithShuttleDebugPrintf ++ // enable debug printf
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(4) ++
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new chipyard.config.AbstractConfig)

class CUTEv3Shuttle512D512V512M256S1CoreConfig extends Config(
//   new cute.WithCUTE(Seq(0,1,2,3)) ++
  new cute.WithCuteCoustomParams(CoustomCuteParam = Cutev3Params.CUTE_8Tops_128SCP) ++
  new freechips.rocketchip.subsystem.WithNBitMemoryBus(256) ++
  new WithNBanks(4) ++
  new freechips.rocketchip.subsystem.WithInclusiveCache(capacityKB=512,outerLatencyCycles=40) ++
  new cutev3.shuttle.WithShuttleTensorUnit(vLen = 512, dLen = 512, VectorParams.refParams, mLen = Option(512),cores = Seq(0)) ++
  new cutev3.common.WithCUTETCM ++
  new cutev3.common.WithCUTESGTCM ++
  new chipyard.config.WithSystemBusWidth(256) ++
  // new shuttle.common.WithShuttleDebugROB ++ // enable debug ROB
  // new shuttle.common.WithShuttleDebugPrintf ++ // enable debug printf
  new shuttle.common.WithShuttleTileBeatBytes(64) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
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

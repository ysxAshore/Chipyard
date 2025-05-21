package chipyard

import org.chipsalliance.cde.config.{Config}
import saturn.common.{VectorParams}
// ---------------------
// BOOM V3 Configs
// Performant, stable baseline
// ---------------------


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

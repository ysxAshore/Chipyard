package chipyard.fpga.yjpvivadoip

import sys.process._

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{SystemBusKey, PeripheryBusKey, ControlBusKey, ExtMem}
import freechips.rocketchip.devices.debug.{DebugModuleKey, ExportDebug, JTAG}
import freechips.rocketchip.devices.tilelink.{DevNullParams, BootROMLocated}
import freechips.rocketchip.diplomacy.{RegionType, AddressSet}
import freechips.rocketchip.resources.{DTSModel, DTSTimebase}
import freechips.rocketchip.util.{SystemFileName}

import sifive.blocks.devices.spi.{PeripherySPIKey, SPIParams}
import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTParams}

import sifive.fpgashells.shell.{DesignKey, JTAGDebugOverlayKey}
import sifive.fpgashells.shell.xilinx.{VCU118ShellPMOD, VCU118DDRSize}

import testchipip.serdes.{SerialTLKey}

import chipyard._
import chipyard.harness._

import freechips.rocketchip.subsystem.ExtBus
import freechips.rocketchip.devices.tilelink.CLINTConsts.size

class WithSystemModifications extends Config((site, here, up) => {
  case DTSTimebase => BigInt((1e6).toLong)
  case BootROMLocated(x) => up(BootROMLocated(x), site).map { p =>
    // invoke makefile for sdboot
    val freqMHz = (site(SystemBusKey).dtsFrequency.get / (1000 * 1000)).toLong
    val make = s"make -C fpga/src/main/resources/vcu118/flashboot PBUS_CLK=${freqMHz} bin"
    require (make.! == 0, "Failed to build bootrom")
    p.copy(hang = 0x10000, contentFileName = SystemFileName(s"./fpga/src/main/resources/vcu118/flashboot/build/flashboot.bin"))
  }
  case ExtMem => up(ExtMem, site).map(x => x.copy(master = x.master.copy(size = site(VCU118DDRSize),beatBytes = 16))) // set extmem to DDR size
  case SerialTLKey => Nil // remove serialized tl port
})

// DOC include start: AbstractVCU118 and Rocket
class WithVCU118Tweaks extends Config(
  // clocking
  new chipyard.harness.WithAllClocksFromHarnessClockInstantiator ++
  new chipyard.clocking.WithPassthroughClockGenerator ++
  new chipyard.config.WithUniformBusFrequencies(25) ++
  new WithFPGAFrequency(25) ++ // default 25mhz freq
  new WithFakeSimJTAGDebug ++ //必须有jtag等debug的线出来
  new WithSystemModifications ++ // setup busses, use sdboot bootrom, setup ext. mem. size
  new freechips.rocketchip.subsystem.WithoutTLMonitors ++
  new freechips.rocketchip.subsystem.WithNMemoryChannels(1)
)


class CUTEBoomVCU118Config extends Config(
  new WithFPGAFrequency(25) ++
  new chipyard.config.WithUniformBusFrequencies(25) ++
  new WithYJPVivadoMMIODevicePort ++ //设置了MMIO的地址段,设置了前端总线的位宽
  new WithFakeAxi4Mem(0) ++ //连接出去的假的ddr4，为了将axi4_mem的线实例化，0可以增加额外延迟
  new WithFakeAXIMMIO ++ //连接出去的假的mmio，为了将axi4_mmio的线实例化
  new WithVivadoIPL2FBusAXI4Punchthrough ++ //L2_S_x --> 前端总线，dma总线
  new WithVivadoIPAXI4MMIOPunchthrough ++   //MMIO_M_x --> MMIO设备，各种blockDesign涉及的IP核通过这条总线连接
  new WithVivadoIPAXI4MemPunchthrough ++    //MEM_M_x --> DDR4内存
  new freechips.rocketchip.subsystem.WithCustomSlavePort(data_width = 64, id_bits = 8) ++ //设置访存总线的位宽
//   new freechips.rocketchip.subsystem.WithDefaultSlavePort ++
  new freechips.rocketchip.subsystem.WithNExtTopInterrupts(4) ++
  new WithVCU118Tweaks ++ //基础的配置
  new chipyard.YJPFPGACUTESmallBoomConfig
)

class VCU118CUTEv3Shuttle512D512V512M256S1CoreConfig extends Config(
  new WithFPGAFrequency(25) ++
  new chipyard.config.WithUniformBusFrequencies(25) ++
  new WithYJPVivadoMMIODevicePort ++ //设置了MMIO的地址段,设置了前端总线的位宽
  new WithFakeAxi4Mem(0) ++ //连接出去的假的ddr4，为了将axi4_mem的线实例化，0可以增加额外延迟
  new WithFakeAXIMMIO ++ //连接出去的假的mmio，为了将axi4_mmio的线实例化
  new WithVivadoIPL2FBusAXI4Punchthrough ++ //L2_S_x --> 前端总线，dma总线
  new WithVivadoIPAXI4MMIOPunchthrough ++   //MMIO_M_x --> MMIO设备，各种blockDesign涉及的IP核通过这条总线连接
  new WithVivadoIPAXI4MemPunchthrough ++    //MEM_M_x --> DDR4内存
  new freechips.rocketchip.subsystem.WithCustomSlavePort(data_width = 64, id_bits = 8) ++ //设置访存总线的位宽
//   new freechips.rocketchip.subsystem.WithDefaultSlavePort ++
  new freechips.rocketchip.subsystem.WithNExtTopInterrupts(4) ++
  new WithVCU118Tweaks ++ //基础的配置
  new chipyard.CUTEv3Shuttle512D512V512M256S1CoreConfig
)

class VCU118CUTEv3Shuttle512D512V512M256S4CoreConfig extends Config(
  new WithFPGAFrequency(25) ++
  new chipyard.config.WithUniformBusFrequencies(25) ++
  new WithYJPVivadoMMIODevicePort ++ //设置了MMIO的地址段,设置了前端总线的位宽
  new WithFakeAxi4Mem(0) ++ //连接出去的假的ddr4，为了将axi4_mem的线实例化，0可以增加额外延迟
  new WithFakeAXIMMIO ++ //连接出去的假的mmio，为了将axi4_mmio的线实例化
  new WithVivadoIPL2FBusAXI4Punchthrough ++ //L2_S_x --> 前端总线，dma总线
  new WithVivadoIPAXI4MMIOPunchthrough ++   //MMIO_M_x --> MMIO设备，各种blockDesign涉及的IP核通过这条总线连接
  new WithVivadoIPAXI4MemPunchthrough ++    //MEM_M_x --> DDR4内存
  new freechips.rocketchip.subsystem.WithCustomSlavePort(data_width = 64, id_bits = 8) ++ //设置访存总线的位宽
//   new freechips.rocketchip.subsystem.WithDefaultSlavePort ++
  new freechips.rocketchip.subsystem.WithNExtTopInterrupts(4) ++
  new WithVCU118Tweaks ++ //基础的配置
  new chipyard.CUTEv3Shuttle512D512V512M256S4CoreConfig
)

class WithFPGAFrequency(fMHz: Double) extends Config(
  new chipyard.harness.WithHarnessBinderClockFreqMHz(fMHz) ++
  new chipyard.config.WithSystemBusFrequency(fMHz) ++
  new chipyard.config.WithPeripheryBusFrequency(fMHz) ++
  new chipyard.config.WithControlBusFrequency(fMHz) ++
  new chipyard.config.WithFrontBusFrequency(fMHz) ++
  new chipyard.config.WithMemoryBusFrequency(fMHz)
)

class WithFPGAFreq25MHz extends WithFPGAFrequency(25)
class WithFPGAFreq50MHz extends WithFPGAFrequency(50)
class WithFPGAFreq75MHz extends WithFPGAFrequency(75)
class WithFPGAFreq100MHz extends WithFPGAFrequency(100)

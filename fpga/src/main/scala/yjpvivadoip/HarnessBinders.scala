package chipyard.fpga.yjpvivadoip

import chisel3._
import chisel3.experimental.{BaseModule}

import org.chipsalliance.diplomacy.nodes.{HeterogeneousBag}
import freechips.rocketchip.tilelink.{TLBundle}

import sifive.blocks.devices.uart.{UARTPortIO}
import sifive.blocks.devices.spi.{HasPeripherySPI, SPIPortIO}
import freechips.rocketchip.devices.debug.{SimJTAG}

import freechips.rocketchip.jtag.{JTAGIO}

import freechips.rocketchip.system._

import testchipip.dram._

import chipyard._
import chipyard.harness._
import chipyard.iobinders._
import sifive.blocks.devices.gpio.GPIOCtrlRegs.port


import chisel3._
import chisel3.util._
import chisel3.reflect.DataMirror
import chisel3.experimental.Direction

import org.chipsalliance.cde.config.{Field, Config, Parameters}
import freechips.rocketchip.diplomacy.{LazyModule, LazyModuleImpLike}
import freechips.rocketchip.system.{SimAXIMem}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.util._
import freechips.rocketchip.jtag.{JTAGIO}
import freechips.rocketchip.devices.debug.{SimJTAG}
import chipyard.iocell._
import testchipip.dram.{SimDRAM}
import testchipip.tsi.{SimTSI, SerialRAM, TSI, TSIIO}
import testchipip.soc.{TestchipSimDTM}
import testchipip.spi.{SimSPIFlashModel}
import testchipip.uart.{UARTAdapter, UARTToSerial}
import testchipip.serdes._
import testchipip.iceblk.{SimBlockDevice, BlockDeviceModel}
import testchipip.cosim.{SpikeCosim}
import icenet.{NicLoopback, SimNetwork}
import chipyard._
import chipyard.clocking.{HasChipyardPRCI}
import chipyard.iobinders._

import chisel3._
import chisel3.reflect.DataMirror
import chisel3.experimental.Analog

import org.chipsalliance.cde.config._
import org.chipsalliance.diplomacy._
import org.chipsalliance.diplomacy.nodes._
import org.chipsalliance.diplomacy.aop._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.diplomacy.bundlebridge._
import freechips.rocketchip.diplomacy.{Resource, ResourceBinding, ResourceAddress, RegionType}
import freechips.rocketchip.devices.debug._
import freechips.rocketchip.jtag.{JTAGIO}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.system.{SimAXIMem}
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4SlaveNode, AXI4MasterNode, AXI4EdgeParameters}
import freechips.rocketchip.util._
import freechips.rocketchip.prci._
import freechips.rocketchip.groundtest.{GroundTestSubsystemModuleImp, GroundTestSubsystem}
import freechips.rocketchip.tilelink.{TLBundle}

import sifive.blocks.devices.gpio._
import sifive.blocks.devices.uart._
import sifive.blocks.devices.spi._
import sifive.blocks.devices.i2c._
import tracegen.{TraceGenSystemModuleImp}

import chipyard.iocell._

import testchipip.serdes.{CanHavePeripheryTLSerial, SerialTLKey}
import testchipip.spi.{SPIChipIO}
import testchipip.boot.{CanHavePeripheryCustomBootPin}
import testchipip.soc.{CanHavePeripheryChipIdPin, CanHaveSwitchableOffchipBus}
import testchipip.util.{ClockedIO}
import testchipip.iceblk.{CanHavePeripheryBlockDevice, BlockDeviceKey, BlockDeviceIO}
import testchipip.cosim.{CanHaveTraceIO, TraceOutputTop, SpikeCosimConfig}
import testchipip.tsi.{CanHavePeripheryUARTTSI, UARTTSIIO}
import icenet.{CanHavePeripheryIceNIC, SimNetwork, NicLoopback, NICKey, NICIOvonly}
import chipyard.{CanHaveMasterTLMemPort, ChipyardSystem, ChipyardSystemModule}
import chipyard.example.{CanHavePeripheryGCD}

case object HarnessBinders extends Field[HarnessBinderFunction]({case _ => })


class WithFakeSimJTAGDebug extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: JTAGPort, chipId: Int) => {
    val dtm_success = WireInit(false.B)
    // when (dtm_success) { th.success := true.B }
    val jtag_wire = Wire(new JTAGIO)
    jtag_wire.TDO.data := port.io.TDO
    jtag_wire.TDO.driven := true.B
    port.io.TCK := jtag_wire.TCK
    port.io.TMS := jtag_wire.TMS
    port.io.TDI := jtag_wire.TDI
    port.io.reset.foreach(_ := th.harnessBinderReset.asBool)
    val jtag = Module(new SimJTAG(tickDelay=3))
    jtag.connect(jtag_wire, th.harnessBinderClock, th.harnessBinderReset.asBool, ~(th.harnessBinderReset.asBool), dtm_success)
  }
})

class WithFakeAxi4Mem(additionalLatency: Int = 0) extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: AXI4MemPort, chipId: Int) => {
    // TODO FIX: This currently makes each SimDRAM contain the entire memory space
    val memSize = port.params.master.size
    val memBase = port.params.master.base
    val lineSize = 64 // cache block size
    val clockFreq = port.clockFreqMHz
    val mem = Module(new SimDRAM(memSize, lineSize, clockFreq, memBase, port.edge.bundle, chipId)).suggestName("fake_dram")

    mem.io.clock := port.io.clock
    mem.io.reset := th.harnessBinderReset.asAsyncReset
    mem.io.axi <> port.io.bits
    // Bug in Chisel implementation. See https://github.com/chipsalliance/chisel3/pull/1781
    def Decoupled[T <: Data](irr: IrrevocableIO[T]): DecoupledIO[T] = {
      require(DataMirror.directionOf(irr.bits) == Direction.Output, "Only safe to cast produced Irrevocable bits to Decoupled.")
      val d = Wire(new DecoupledIO(chiselTypeOf(irr.bits)))
      d.bits := irr.bits
      d.valid := irr.valid
      irr.ready := d.ready
      d
    }
    if (additionalLatency > 0) {
      withClock (port.io.clock) {
        mem.io.axi.aw  <> (0 until additionalLatency).foldLeft(Decoupled(port.io.bits.aw))((t, _) => Queue(t, 1, pipe=true))
        mem.io.axi.w   <> (0 until additionalLatency).foldLeft(Decoupled(port.io.bits.w ))((t, _) => Queue(t, 1, pipe=true))
        port.io.bits.b <> (0 until additionalLatency).foldLeft(Decoupled(mem.io.axi.b   ))((t, _) => Queue(t, 1, pipe=true))
        mem.io.axi.ar  <> (0 until additionalLatency).foldLeft(Decoupled(port.io.bits.ar))((t, _) => Queue(t, 1, pipe=true))
        port.io.bits.r <> (0 until additionalLatency).foldLeft(Decoupled(mem.io.axi.r   ))((t, _) => Queue(t, 1, pipe=true))
      }
    }
  }
})

class WithFakeAXIMMIO extends HarnessBinder({
  case (th: HasHarnessInstantiators, port: AXI4MMIOPort, chipId: Int) => {
    val mmio_mem = LazyModule(new SimAXIMem(port.edge, size = port.params.size)(Parameters.empty))
    withClock(port.io.clock) { Module(mmio_mem.module).suggestName("fake_mmio_mem") }
    mmio_mem.io_axi4.head <> port.io.bits
  }
})

class WithYJPVivadoMMIODevicePort extends Config((site, here, up) => {
  case ExtBus => {
    //scala输出beatBytes
    val beatBytes = site(freechips.rocketchip.subsystem.MemoryBusKey).beatBytes
    Some(freechips.rocketchip.subsystem.MasterPortParams(
                      base = 0x60000000L,
                      size = 0x04000000L,
                      beatBytes = site(freechips.rocketchip.subsystem.MemoryBusKey).beatBytes,
                      idBits = 4))
                    }
})

class WithVivadoIPAXI4MemPunchthrough extends OverrideLazyIOBinder({
  (system: CanHaveMasterAXI4MemPort) => {
    implicit val p: Parameters = GetSystemParameters(system)
    val clockSinkNode = p(ExtMem).map(_ => ClockSinkNode(Seq(ClockSinkParameters())))
    clockSinkNode.map(_ := system.asInstanceOf[HasTileLinkLocations].locateTLBusWrapper(MBUS).fixedClockNode)
    def clockBundle = clockSinkNode.get.in.head._1

    InModuleBody {
      val ports: Seq[AXI4MemPort] = system.mem_axi4.zipWithIndex.map({ case (m, i) =>
        val port = IO(new ClockedIO(DataMirror.internal.chiselTypeClone[AXI4Bundle](m))).suggestName(s"Mem_M_${i}")
        port.bits <> m
        port.clock := clockBundle.clock
        AXI4MemPort(() => port, p(ExtMem).get, system.memAXI4Node.edges.in(i), p(MemoryBusKey).dtsFrequency.get.toInt)
      }).toSeq
      (ports, Nil)
    }
  }
})


class WithVivadoIPAXI4MMIOPunchthrough extends OverrideLazyIOBinder({
  (system: CanHaveMasterAXI4MMIOPort) => {
    implicit val p: Parameters = GetSystemParameters(system)
    val clockSinkNode = p(ExtBus).map(_ => ClockSinkNode(Seq(ClockSinkParameters())))
    clockSinkNode.map(_ := system.asInstanceOf[HasTileLinkLocations].locateTLBusWrapper(SBUS).fixedClockNode)
    def clockBundle = clockSinkNode.get.in.head._1

    InModuleBody {
      val ports: Seq[AXI4MMIOPort] = system.mmio_axi4.zipWithIndex.map({ case (m, i) =>
        val port = IO(new ClockedIO(DataMirror.internal.chiselTypeClone[AXI4Bundle](m))).suggestName(s"MMIO_M_${i}")
        port.bits <> m
        port.clock := clockBundle.clock
        AXI4MMIOPort(() => port, p(ExtBus).get, system.mmioAXI4Node.edges.in(i))
      }).toSeq
      (ports, Nil)
    }
  }
})

class WithVivadoIPL2FBusAXI4Punchthrough extends OverrideLazyIOBinder({
  (system: CanHaveSlaveAXI4Port) => {
    implicit val p: Parameters = GetSystemParameters(system)
    val clockSinkNode = p(ExtIn).map(_ => ClockSinkNode(Seq(ClockSinkParameters())))
    val fbus = system.asInstanceOf[HasTileLinkLocations].locateTLBusWrapper(FBUS)
    clockSinkNode.map(_ := fbus.fixedClockNode)
    def clockBundle = clockSinkNode.get.in.head._1

    InModuleBody {
      val ports: Seq[AXI4InPort] = system.l2_frontend_bus_axi4.zipWithIndex.map({ case (m, i) =>
        val port = IO(new ClockedIO(Flipped(DataMirror.internal.chiselTypeClone[AXI4Bundle](m)))).suggestName(s"L2_S_${i}")
        m <> port.bits
        port.clock := clockBundle.clock
        AXI4InPort(() => port, p(ExtIn).get)
      }).toSeq
      (ports, Nil)
    }
  }
})
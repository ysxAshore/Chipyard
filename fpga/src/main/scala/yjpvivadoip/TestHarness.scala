package chipyard.fpga.yjpvivadoip

import chisel3._

import freechips.rocketchip.diplomacy.{LazyModule, LazyRawModuleImp, BundleBridgeSource}
import org.chipsalliance.cde.config.{Parameters}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.diplomacy.{IdRange, TransferSizes}
import freechips.rocketchip.subsystem.{SystemBusKey}
import freechips.rocketchip.prci._
import sifive.fpgashells.shell.xilinx._
import sifive.fpgashells.ip.xilinx.{IBUF, PowerOnResetFPGAOnly}
import sifive.fpgashells.shell._
import sifive.fpgashells.clocks._
import sifive.fpgashells.yjpshell.xilinx._

import sifive.blocks.devices.uart.{PeripheryUARTKey, UARTPortIO}
import sifive.blocks.devices.spi.{PeripherySPIKey, SPIPortIO}

import chipyard._
import chipyard.harness._
import freechips.rocketchip.subsystem.CanHaveMasterAXI4MMIOPort
import freechips.rocketchip.amba.axi4.AXI4MasterNode
import freechips.rocketchip.amba.axi4._
import freechips.rocketchip.amba.axis.AXISMasterNode


class YJPVivadoIPHarness(override implicit val p: Parameters) extends YJPVCU118IPBasicOverlays {

  def dp = designParameters

  val dutFreqMHz = (dp(SystemBusKey).dtsFrequency.get / (1000 * 1000)).toInt
  println(s"VCU118 FPGA Base Clock Freq: ${dutFreqMHz} MHz")
  
  override lazy val module = new YJPVivadoIPHarnessImp(this)
}

class YJPVivadoIPHarnessImp(_outer: YJPVivadoIPHarness) extends LazyRawModuleImp(_outer) with HasHarnessInstantiators {
  override def provideImplicitClockToLazyChildren = true
  val vcu118Outer = _outer

  val reset = IO(Input(Bool())).suggestName("reset")
  val clock = IO(Input(Clock())).suggestName("clock")

  def referenceClockFreqMHz = _outer.dutFreqMHz
  def referenceClock = clock
  def referenceReset = reset
  def success = { require(false, "Unused"); false.B }

  childClock := referenceClock
  childReset := referenceReset

  println("Harness clock/reset initialized.")
  instantiateChipTops()
}

package sifive.fpgashells.yjpshell.xilinx

import chisel3._
import chisel3.experimental.{Analog, attach}
import chisel3.experimental.dataview._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.prci._
import org.chipsalliance.cde.config._
import sifive.fpgashells.clocks._
import sifive.fpgashells.devices.xilinx.xdma._
import sifive.fpgashells.devices.xilinx.xilinxvcu118mig._
import sifive.fpgashells.ip.xilinx._
import sifive.fpgashells.ip.xilinx.xxv_ethernet._
import sifive.fpgashells.ip.xilinx.vcu118mig._
import sifive.fpgashells.shell._
import sifive.fpgashells.shell.xilinx._

abstract class YJPVCU118IPBasicOverlays()(implicit p: Parameters) extends Shell{

  //val spi_flash = Overlay(SPIFlashOverlayKey, new SPIFlashVCU118ShellPlacer(this, SPIFlashShellInput()))
  //SPI Flash not functional
}

// class VCU118Shell()(implicit p: Parameters) extends VCU118ShellBasicOverlays
// {
//   val pmod_is_sdio  = p(VCU118ShellPMOD) == "SDIO"
//   val pmod_j53_is_jtag = p(VCU118ShellPMOD2) == "PMODJ53_JTAG"
//   val jtag_location = Some(if (pmod_is_sdio) (if (pmod_j53_is_jtag) "PMOD_J53" else "FMC_J2") else "PMOD_J52")

//   // Order matters; ddr depends on sys_clock
//   val uart      = Overlay(UARTOverlayKey, new UARTVCU118ShellPlacer(this, UARTShellInput()))
//   val sdio      = if (pmod_is_sdio) Some(Overlay(SPIOverlayKey, new SDIOVCU118ShellPlacer(this, SPIShellInput()))) else None
//   val jtag      = Overlay(JTAGDebugOverlayKey, new JTAGDebugVCU118ShellPlacer(this, JTAGDebugShellInput(location = jtag_location)))
//   val cjtag     = Overlay(cJTAGDebugOverlayKey, new cJTAGDebugVCU118ShellPlacer(this, cJTAGDebugShellInput()))
//   val jtagBScan = Overlay(JTAGDebugBScanOverlayKey, new JTAGDebugBScanVCU118ShellPlacer(this, JTAGDebugBScanShellInput()))
//   val fmc       = Overlay(PCIeOverlayKey, new PCIeVCU118FMCShellPlacer(this, PCIeShellInput()))
//   val edge      = Overlay(PCIeOverlayKey, new PCIeVCU118EdgeShellPlacer(this, PCIeShellInput()))

//   val topDesign = LazyModule(p(DesignKey)(designParameters))

//   // Place the sys_clock at the Shell if the user didn't ask for it
//   designParameters(ClockInputOverlayKey).foreach { unused =>
//     val source = unused.place(ClockInputDesignInput()).overlayOutput.node
//     val sink = ClockSinkNode(Seq(ClockSinkParameters()))
//     sink := source
//   }

//   override lazy val module = new LazyRawModuleImp(this) {
//     val reset = IO(Input(Bool()))
//     xdc.addPackagePin(reset, "L19")
//     xdc.addIOStandard(reset, "LVCMOS12")

//     val reset_ibuf = Module(new IBUF)
//     reset_ibuf.io.I := reset

//     val sysclk: Clock = sys_clock.get() match {
//       case Some(x: SysClockVCU118PlacedOverlay) => x.clock
//     }

//     val powerOnReset: Bool = PowerOnResetFPGAOnly(sysclk)
//     sdc.addAsyncPath(Seq(powerOnReset))

//     val ereset: Bool = chiplink.get() match {
//       case Some(x: ChipLinkVCU118PlacedOverlay) => !x.ereset_n
//       case _ => false.B
//     }

//     pllReset := (reset_ibuf.io.O || powerOnReset || ereset)
//   }
// }

/*
   Copyright 2016 SiFive, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

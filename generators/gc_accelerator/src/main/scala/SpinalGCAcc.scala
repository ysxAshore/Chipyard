package spinal_gc

import chisel3._
import chisel3.util._

class SpinalGCAcc extends Module {
  val io = IO(new SpinalGCTopIO)

  val acc = Module(new GCTop)

  acc.io.clk <> clock
  acc.io.reset <> reset

  acc.io.io_mmu2llc_Request_ready <> io.mmu2llc.Request.ready

  io.mmu2llc.Request.valid <> acc.io.io_mmu2llc_Request_valid
  io.mmu2llc.Request.bits.RequestVirtualAddr <> acc.io.io_mmu2llc_Request_payload_RequestVirtualAddr
  io.mmu2llc.Request.bits.RequestSourceID <> acc.io.io_mmu2llc_Request_payload_RequestSourceID
  io.mmu2llc.Request.bits.RequestType_isWrite <> acc.io.io_mmu2llc_Request_payload_RequestType_isWrite
  io.mmu2llc.Request.bits.RequestData <> acc.io.io_mmu2llc_Request_payload_RequestData
  io.mmu2llc.Request.bits.RequestWStrb <> acc.io.io_mmu2llc_Request_payload_RequestWStrb

  acc.io.io_mmu2llc_ConherentRequsetSourceID_valid <> io.mmu2llc.ConherentRequsetSourceID.valid
  acc.io.io_mmu2llc_ConherentRequsetSourceID_payload <> io.mmu2llc.ConherentRequsetSourceID.bits

  acc.io.io_mmu2llc_Response_payload_ResponseData <> io.mmu2llc.Response.bits.ReseponseData
  acc.io.io_mmu2llc_Response_valid <> io.mmu2llc.Response.valid
  acc.io.io_mmu2llc_Response_payload_ResponseSourceID <> io.mmu2llc.Response.bits.ReseponseSourceID

  io.mmu2llc.Response.ready <> acc.io.io_mmu2llc_Response_ready

  io.ctrl2top.Done <> acc.io.io_ctrl2top_Done
  io.ctrl2top.Ready <> acc.io.io_ctrl2top_Ready

  acc.io.io_ctrl2top_Valid <> io.ctrl2top.Valid
  acc.io.io_ctrl2top_RegionAttrBase <> io.ctrl2top.RegionAttrBase
  acc.io.io_ctrl2top_RegionAttrBiasedBase <> io.ctrl2top.RegionAttrBiasedBase
  acc.io.io_ctrl2top_RegionAttrShiftBy <> io.ctrl2top.RegionAttrShiftBy
  acc.io.io_ctrl2top_HeapRegionBias <> io.ctrl2top.HeapRegionBias
  acc.io.io_ctrl2top_HeapRegionShiftBy <> io.ctrl2top.HeapRegionShiftBy
  acc.io.io_ctrl2top_HeapRegionBiasedBase <> io.ctrl2top.HeapRegionBiasedBase
  acc.io.io_ctrl2top_HumongousReclaimCandidatesBoolBase <> io.ctrl2top.HumongousReclaimCandidatesBoolBase
  acc.io.io_ctrl2top_ParScanThreadStatePtr <> io.ctrl2top.ParScanThreadStatePtr
  acc.io.io_ctrl2top_TaskQueue_BottomAddr <> io.ctrl2top.TaskQueue_BottomAddr
  acc.io.io_ctrl2top_TaskQueue_AgeTopAddr <> io.ctrl2top.TaskQueue_AgeTopAddr
  acc.io.io_ctrl2top_TaskQueue_ElemsBase <> io.ctrl2top.TaskQueue_ElemsBase
}

class GCTop extends BlackBox with HasBlackBoxPath{
  val io = IO(new GCTopIO)
  addPath(s"/home/sxyang/Projects/java_gc/hwgc_spinal/sim/vsrc/GCTop.v")
}

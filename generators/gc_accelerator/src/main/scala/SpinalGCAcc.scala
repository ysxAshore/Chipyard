package spinal_gc

import chisel3._
import chisel3.util._

class SpinalGCAcc extends Module {
  val io = IO(new SpinalGCTopIO)
  val acc = Module(new GCTop)

  acc.io.clk := clock
  acc.io.reset := reset

  acc.io.io_mmu2llc_Request_ready := io.mmu2llc.Request.ready

  io.mmu2llc.Request.valid := acc.io.io_mmu2llc_Request_valid
  io.mmu2llc.Request.bits.RequestSourceID := acc.io.io_mmu2llc_Request_payload_RequestSourceID
  io.mmu2llc.Request.bits.RequestVirtualAddr := acc.io.io_mmu2llc_Request_payload_RequestVirtualAddr
  io.mmu2llc.Request.bits.RequestType_isWrite := acc.io.io_mmu2llc_Request_payload_RequestType_isWrite
  io.mmu2llc.Request.bits.RequestData := acc.io.io_mmu2llc_Request_payload_RequestData
  io.mmu2llc.Request.bits.RequestWStrb := acc.io.io_mmu2llc_Request_payload_RequestWStrb

  acc.io.io_mmu2llc_ConherentRequsetSourceID_valid := io.mmu2llc.ConherentRequsetSourceID.valid
  acc.io.io_mmu2llc_ConherentRequsetSourceID_payload := io.mmu2llc.ConherentRequsetSourceID.bits

  acc.io.io_mmu2llc_Response_valid := io.mmu2llc.Response.valid
  acc.io.io_mmu2llc_Response_payload_ResponseData := io.mmu2llc.Response.bits.ReseponseData
  acc.io.io_mmu2llc_Response_payload_ResponseSourceID := io.mmu2llc.Response.bits.ReseponseSourceID

  io.mmu2llc.Response.ready := acc.io.io_mmu2llc_Response_ready

  io.ctrl2top.Done := acc.io.io_ctrl2top_Done
  io.ctrl2top.ctrl.ready := acc.io.io_ctrl2top_cmd_ready

  acc.io.io_ctrl2top_cmd_valid := io.ctrl2top.ctrl.valid
  acc.io.io_ctrl2top_cmd_payload_ChunkSize := io.ctrl2top.ctrl.bits.ChunkSize
  acc.io.io_ctrl2top_cmd_payload_CardTablePtr := io.ctrl2top.ctrl.bits.CardTablePtr
  acc.io.io_ctrl2top_cmd_payload_AgeThreshold := io.ctrl2top.ctrl.bits.AgeThreshold
  acc.io.io_ctrl2top_cmd_payload_StepperOffset := io.ctrl2top.ctrl.bits.StepperOffset
  acc.io.io_ctrl2top_cmd_payload_YoungWordsBase := io.ctrl2top.ctrl.bits.YoungWordsBase
  acc.io.io_ctrl2top_cmd_payload_RegionAttrBase := io.ctrl2top.ctrl.bits.RegionAttrBase
  acc.io.io_ctrl2top_cmd_payload_HeapRegionBias := io.ctrl2top.ctrl.bits.HeapRegionBias
  acc.io.io_ctrl2top_cmd_payload_PlabAllocatorPtr := io.ctrl2top.ctrl.bits.PlabAllocatorPtr
  acc.io.io_ctrl2top_cmd_payload_RegionAttrShiftBy := io.ctrl2top.ctrl.bits.RegionAttrShiftBy
  acc.io.io_ctrl2top_cmd_payload_HeapRegionShiftBy := io.ctrl2top.ctrl.bits.HeapRegionShiftBy
  acc.io.io_ctrl2top_cmd_payload_LogOfHRGrainBytes := io.ctrl2top.ctrl.bits.LogOfHRGrainBytes
  acc.io.io_ctrl2top_cmd_payload_RegionAttrBiasedBase := io.ctrl2top.ctrl.bits.RegionAttrBiasedBase
  acc.io.io_ctrl2top_cmd_payload_HeapRegionBiasedBase := io.ctrl2top.ctrl.bits.HeapRegionBiasedBase
  acc.io.io_ctrl2top_cmd_payload_ParScanThreadStatePtr := io.ctrl2top.ctrl.bits.ParScanThreadStatePtr
  acc.io.io_ctrl2top_cmd_payload_TaskQueue_Bottom := io.ctrl2top.ctrl.bits.TaskQueue_Bottom
  acc.io.io_ctrl2top_cmd_payload_TaskQueue_ElemsBase := io.ctrl2top.ctrl.bits.TaskQueue_ElemsBase
  acc.io.io_ctrl2top_cmd_payload_HumongousReclaimCandidatesBoolBase := io.ctrl2top.ctrl.bits.HumongousReclaimCandidatesBoolBase
  acc.io.io_ctrl2top_cmd_payload_G1h := io.ctrl2top.ctrl.bits.G1h
  acc.io.io_ctrl2top_cmd_payload_IntArrayKlassObj := io.ctrl2top.ctrl.bits.IntArrayKlassObj
  acc.io.io_ctrl2top_cmd_payload_ObjectKlass := io.ctrl2top.ctrl.bits.ObjectKlass
  acc.io.io_ctrl2top_cmd_payload_LockPtr := io.ctrl2top.ctrl.bits.LockPtr
  acc.io.io_ctrl2top_cmd_payload_Thread := io.ctrl2top.ctrl.bits.Thread
  acc.io.io_ctrl2top_cmd_payload_DummyRegion := io.ctrl2top.ctrl.bits.DummyRegion
  acc.io.io_ctrl2top_cmd_payload_CompressedOopBase := io.ctrl2top.ctrl.bits.CompressedOopBase
  acc.io.io_ctrl2top_cmd_payload_CompressedKlassPointerBase := io.ctrl2top.ctrl.bits.CompressedKlassPointerBase
  acc.io.io_ctrl2top_cmd_payload_CompressedFlag := io.ctrl2top.ctrl.bits.CompressedFlag
}

class GCTop extends BlackBox with HasBlackBoxPath{
  val io = IO(new GCTopIO)
  addPath(s"/home/sxyang/Projects/java_gc/hwgc_spinal/sim/vsrc/GCTop.v")
}

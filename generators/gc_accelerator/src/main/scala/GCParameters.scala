package spinal_gc

import chisel3._
import chisel3.util._
import freechips.rocketchip.tile.LazyRoCC //this needs delete?
import org.chipsalliance.cde.config.{Field, Parameters} //this needs delete?

case class GCAccConfig()
case object GCAccKey extends Field[Option[GCAccConfig]](None)
case object BuildDMAInterface extends Field[Seq[Parameters => LazyRoCC]](Nil)

trait HWParameters {
  val DebugEnable = true

  val MMUAddrWidth = 64
  val MMUDataWidth = 64

  val SourceMaxNum = 64
  val SourceMaxNumBitSize = log2Ceil(SourceMaxNum) + 1
}

class MMU2TLIO extends Bundle with HWParameters{

  //发出的访存请求
  val Request = Flipped(DecoupledIO(new Bundle{
    val RequestVirtualAddr = UInt(MMUAddrWidth.W)
    val RequestData = UInt(MMUDataWidth.W)
    val RequestSourceID = UInt(SourceMaxNumBitSize.W)
    val RequestType_isWrite = Bool()
    val RequestWStrb = UInt((MMUDataWidth / 8).W)
  }))
  //读请求分发到的TL Link的事务编号
  val ConherentRequsetSourceID = Valid(UInt(SourceMaxNumBitSize.W))

  //Memoryloader一定能保证收回！
  val Response = DecoupledIO(new Bundle{
    val ReseponseData = UInt(MMUDataWidth.W)
    val ReseponseSourceID = UInt(SourceMaxNumBitSize.W)
  })
}

class Ctrl2Top extends Bundle with HWParameters {
  val ChunkSize = Input(UInt(32.W))
  val CardTablePtr = Input(UInt(MMUAddrWidth.W))
  val AgeThreshold = Input(UInt(32.W))
  val StepperOffset = Input(UInt(MMUDataWidth.W))
  val YoungWordsBase = Input(UInt(MMUAddrWidth.W))
  val RegionAttrBase = Input(UInt(MMUAddrWidth.W))
  val HeapRegionBias = Input(UInt(32.W))
  val PlabAllocatorPtr = Input(UInt(MMUAddrWidth.W))
  val RegionAttrShiftBy = Input(UInt(32.W))
  val HeapRegionShiftBy = Input(UInt(32.W))
  val LogOfHRGrainBytes = Input(UInt(32.W))
  val RegionAttrBiasedBase = Input(UInt(MMUAddrWidth.W))
  val HeapRegionBiasedBase = Input(UInt(MMUAddrWidth.W))
  val ParScanThreadStatePtr = Input(UInt(MMUAddrWidth.W))
  val TaskQueue_BottomAddr = Input(UInt(MMUAddrWidth.W))
  val TaskQueue_AgeTopAddr = Input(UInt(MMUAddrWidth.W))
  val TaskQueue_ElemsBase = Input(UInt(MMUAddrWidth.W))
  val HumongousReclaimCandidatesBoolBase = Input(UInt(MMUAddrWidth.W))

  val Valid = Input(Bool())
  val Ready = Output(Bool())
  val Done = Output(Bool())
}

class SpinalGCTopIO extends Bundle{
  val mmu2llc = Flipped(new MMU2TLIO)
  val ctrl2top = Flipped(new Ctrl2Top)
}

class GCTopIO extends Bundle{
  val io_mmu2llc_Request_valid = Output(Bool())
  val io_mmu2llc_Request_ready = Input(Bool())
  val io_mmu2llc_Request_payload_RequestVirtualAddr = Output(UInt(64.W))
  val io_mmu2llc_Request_payload_RequestData = Output(UInt(64.W))
  val io_mmu2llc_Request_payload_RequestSourceID = Output(UInt(7.W))
  val io_mmu2llc_Request_payload_RequestType_isWrite = Output(Bool())
  val io_mmu2llc_Request_payload_RequestWStrb = Output(UInt(8.W))
  val io_mmu2llc_ConherentRequsetSourceID_valid = Input(Bool())
  val io_mmu2llc_ConherentRequsetSourceID_payload = Input(UInt(7.W))
  val io_mmu2llc_Response_valid = Input(Bool())
  val io_mmu2llc_Response_ready = Output(Bool())
  val io_mmu2llc_Response_payload_ResponseData = Input(UInt(64.W))
  val io_mmu2llc_Response_payload_ResponseSourceID = Input(UInt(7.W))
  val io_ctrl2top_ChunkSize = Input(UInt(32.W))
  val io_ctrl2top_CardTablePtr = Input(UInt(64.W))
  val io_ctrl2top_AgeThreshold = Input(UInt(32.W))
  val io_ctrl2top_StepperOffset = Input(UInt(64.W))
  val io_ctrl2top_YoungWordsBase = Input(UInt(64.W))
  val io_ctrl2top_RegionAttrBase = Input(UInt(64.W))
  val io_ctrl2top_HeapRegionBias = Input(UInt(32.W))
  val io_ctrl2top_PlabAllocatorPtr = Input(UInt(64.W))
  val io_ctrl2top_RegionAttrShiftBy = Input(UInt(32.W))
  val io_ctrl2top_HeapRegionShiftBy = Input(UInt(32.W))
  val io_ctrl2top_LogOfHRGrainBytes = Input(UInt(32.W))
  val io_ctrl2top_RegionAttrBiasedBase = Input(UInt(64.W))
  val io_ctrl2top_HeapRegionBiasedBase = Input(UInt(64.W))
  val io_ctrl2top_ParScanThreadStatePtr = Input(UInt(64.W))
  val io_ctrl2top_TaskQueue_BottomAddr = Input(UInt(64.W))
  val io_ctrl2top_TaskQueue_AgeTopAddr = Input(UInt(64.W))
  val io_ctrl2top_TaskQueue_ElemsBase = Input(UInt(64.W))
  val io_ctrl2top_HumongousReclaimCandidatesBoolBase = Input(UInt(64.W))
  val io_ctrl2top_Valid = Input(Bool())
  val io_ctrl2top_Ready = Output(Bool())
  val io_ctrl2top_Done = Output(Bool())
  val clk = Input(Clock())
  val reset = Input(Bool())
}

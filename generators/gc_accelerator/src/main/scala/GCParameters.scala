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

  val GCElementWidth = 64

  val MMUAddrWidth = 64
  val MMUDataWidth = 256

  val SourceMaxNum = 32
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

case class Ctrl2TopPayload() extends Bundle with HWParameters {
  val ChunkSize = UInt(32.W)
  val AgeThreshold = UInt(32.W)
  val HeapRegionBias = UInt(32.W)
  val RegionAttrShiftBy = UInt(32.W)
  val HeapRegionShiftBy = UInt(32.W)
  val LogOfHRGrainBytes = UInt(32.W)
  val StepperOffset = UInt(GCElementWidth.W)
  val YoungWordsBase = UInt(GCElementWidth.W)
  val RegionAttrBase = UInt(GCElementWidth.W)
  val PlabAllocatorPtr = UInt(GCElementWidth.W)
  val RegionAttrBiasedBase = UInt(GCElementWidth.W)
  val HeapRegionBiasedBase = UInt(GCElementWidth.W)
  val ParScanThreadStatePtr = UInt(GCElementWidth.W)
  val TaskQueue_Bottom = UInt(32.W)
  val TaskQueue_ElemsBase = UInt(GCElementWidth.W)
  val HumongousReclaimCandidatesBoolBase = UInt(GCElementWidth.W)
  val CardTablePtr = UInt(GCElementWidth.W)
  val G1h = UInt(GCElementWidth.W)
  val IntArrayKlassObj = UInt(GCElementWidth.W)
  val ObjectKlass = UInt(GCElementWidth.W)
  val LockPtr = UInt(GCElementWidth.W)
  val Thread = UInt(GCElementWidth.W)
  val DummyRegion = UInt(GCElementWidth.W)
  val CompressedOopBase = UInt(GCElementWidth.W)
  val CompressedKlassPointerBase = UInt(GCElementWidth.W)
  val CompressedFlag = UInt(32.W)
}


case class Ctrl2Top() extends Bundle with HWParameters {
  val ctrl = Flipped(DecoupledIO(Ctrl2TopPayload()))
  val Done = Output(Bool())
}


class SpinalGCTopIO extends Bundle{
  val mmu2llc = Flipped(new MMU2TLIO)
  val ctrl2top = new Ctrl2Top
}

class GCTopIO extends Bundle{
  val io_mmu2llc_Request_valid = Output(Bool())
  val io_mmu2llc_Request_ready = Input(Bool())
  val io_mmu2llc_Request_payload_RequestVirtualAddr = Output(UInt(64.W))
  val io_mmu2llc_Request_payload_RequestData = Output(UInt(256.W))
  val io_mmu2llc_Request_payload_RequestSourceID = Output(UInt(6.W))
  val io_mmu2llc_Request_payload_RequestType_isWrite = Output(Bool())
  val io_mmu2llc_Request_payload_RequestWStrb = Output(UInt(32.W))
  val io_mmu2llc_Request_payload_RequestSize = Output(UInt(6.W))
  val io_mmu2llc_Request_payload_NeedDoCmpxChg = Output(Bool())
  val io_mmu2llc_Request_payload_NeedResponse = Output(Bool())
  val io_mmu2llc_ConherentRequsetSourceID_valid = Input(Bool())
  val io_mmu2llc_ConherentRequsetSourceID_payload = Input(UInt(6.W))
  val io_mmu2llc_Response_valid = Input(Bool())
  val io_mmu2llc_Response_ready = Output(Bool())
  val io_mmu2llc_Response_payload_ResponseData = Input(UInt(256.W))
  val io_mmu2llc_Response_payload_ResponseSourceID = Input(UInt(6.W))
  val io_ctrl2top_cmd_payload_ChunkSize = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_AgeThreshold = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_HeapRegionBias = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_RegionAttrShiftBy = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_HeapRegionShiftBy = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_LogOfHRGrainBytes = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_StepperOffset = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_YoungWordsBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_RegionAttrBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_PlabAllocatorPtr = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_RegionAttrBiasedBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_HeapRegionBiasedBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_ParScanThreadStatePtr = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_TaskQueue_Bottom = Input(UInt(32.W))
  val io_ctrl2top_cmd_payload_TaskQueue_ElemsBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_HumongousReclaimCandidatesBoolBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_CardTablePtr = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_G1h = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_IntArrayKlassObj = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_ObjectKlass = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_LockPtr = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_Thread = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_DummyRegion = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_CompressedOopBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_CompressedKlassPointerBase = Input(UInt(64.W))
  val io_ctrl2top_cmd_payload_CompressedFlag = Input(UInt(32.W))
  val io_ctrl2top_cmd_valid = Input(Bool())
  val io_ctrl2top_cmd_ready = Output(Bool())
  val io_ctrl2top_Done = Output(Bool())
  val clk = Input(Clock())
  val reset = Input(Bool())
}

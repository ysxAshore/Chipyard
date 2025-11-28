package spinal_gc

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._ //for rocc
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config._

case object BuildGCAcc extends Field[Parameters => Module]

class WithGCAcc(EnableGCAccList: Seq[Int]) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC, site) ++ EnableGCAccList.collect {
    case tileId if tileId == site(TileKey).tileId =>
      (p: Parameters) => {
        println("[GCAccInterface] TileID: " + tileId)
        val regWidth = 64 // 寄存器位宽
        val gcAcc = LazyModule(new RoCC2GCAcc(OpcodeSet.all)(p))
        gcAcc
      }
  }
  case GCAccKey => true
  case BuildDMAInterface => true
})

class RoCC2GCAcc(opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC(opcodes)
  with HWParameters{
  override lazy val module = new GCAccTile(this)
  lazy val LLCMemPort = LazyModule(new GCAcc2TL)
  tlNode := TLWidthWidget(MMUDataWidth / 8) := LLCMemPort.node
}

class GCAcc2TL(implicit p: Parameters) extends LazyModule with HWParameters{
  lazy val module = new GCAcc2TLImp(this)
  val node = TLClientNode(Seq(TLClientPortParameters(Seq(TLClientParameters(
    name = "GCAcc",
    sourceId = IdRange(0, SourceMaxNum))))))
}

class GCAcc2TLImp(outer: GCAcc2TL) extends LazyModuleImp(outer) with HWParameters{
  val edge = outer.node.edges.out(0)
  val (tl_out, _) = outer.node.out(0)

  val io = IO(new Bundle{
    val mmu = new MMU2TLIO
    val idle = Output(Bool())
  })

  val busy = RegInit(VecInit(Seq.fill(SourceMaxNum)(false.B)))
  val id = WireInit(0.U(SourceMaxNumBitSize.W))

  val is_idle = !busy.reduce(_|_)
  val is_full = busy.reduce(_&_)
  io.idle := is_idle

  for(i <- 0 until SourceMaxNum){
    when(!busy(i)){
      id := i.U
    }
  }

  // default value
  io.mmu.ConherentRequsetSourceID.bits := id
  io.mmu.ConherentRequsetSourceID.valid := !is_full
  io.mmu.Response.bits.ReseponseData := 0.U

  when(!is_full){
    when(tl_out.a.fire){
      busy(id) := true.B
    }
  }.otherwise{
    if(DebugEnable){
      printf("[GCAcc2Interface.node]sourceId queue is_full\n")
    }
  }

  // AccessAck 响应 Put 操作(写)
  // AccessAckData 响应 Get 操作(读)
  // read 需要等待上游是否可以确认接受
  // write 写回的ack直接确认
  tl_out.d.ready := io.mmu.Response.ready
  when(tl_out.d.bits.opcode === TLMessages.AccessAck && tl_out.d.valid && busy(tl_out.d.bits.source)){
    tl_out.d.ready := true.B
  }

  when(tl_out.d.fire){
    busy(tl_out.d.bits.source) := false.B
  }

  tl_out.a.valid := io.mmu.Request.valid && !is_full
  tl_out.a.bits := Mux1H(Seq(
    (io.mmu.Request.bits.RequestType_isWrite === 0.U) -> edge.Get(id, io.mmu.Request.bits.RequestVirtualAddr, log2Ceil(MMUDataWidth / 8).U)._2, // id addr size
    (io.mmu.Request.bits.RequestType_isWrite === 1.U) -> edge.Put(id, io.mmu.Request.bits.RequestVirtualAddr, log2Ceil(MMUDataWidth / 8).U, io.mmu.Request.bits.RequestData, io.mmu.Request.bits.RequestWStrb)._2
  ))

  // 这里read、write都要判断
  io.mmu.Response.valid := tl_out.d.valid && (
    tl_out.d.bits.opcode === TLMessages.AccessAckData ||
    tl_out.d.bits.opcode === TLMessages.AccessAck
  )
  io.mmu.Request.ready := tl_out.a.ready && !is_full
  io.mmu.Response.bits.ReseponseData := tl_out.d.bits.data
  io.mmu.Response.bits.ReseponseSourceID := tl_out.d.bits.source

  val time_stamp = RegInit(0.U(64.W))
  time_stamp := time_stamp + 1.U
  when(io.mmu.Response.fire){
    if (DebugEnable)
    {
      printf("[CUTE2YGJK.node<%d>]io.mmu.Response.fire: %x, io.mmu.Response.bits.ReseponseData: %x\n", time_stamp, io.mmu.Response.bits.ReseponseSourceID, io.mmu.Response.bits.ReseponseData)
    }
  }
  // debug
  when(io.mmu.Request.fire){
    if(DebugEnable){
      printf("[GCAcc2Interface.node<%d>]sourceid: %x, RequestType_isWrite %x, RequestVirtualAddr %x, RequestWStrb %x, RequestData %x\n", time_stamp, io.mmu.ConherentRequsetSourceID.bits, io.mmu.Request.bits.RequestType_isWrite, io.mmu.Request.bits.RequestVirtualAddr, io.mmu.Request.bits.RequestWStrb, io.mmu.Request.bits.RequestData)
    }
  }
}

// custion0~3: 0B 2B 5B 7B
class GCAccTile(outer: RoCC2GCAcc) extends LazyRoCCModuleImp(outer) with HWParameters{
  val acc = Module(new SpinalGCAcc)
  val mem = outer.LLCMemPort.module

  val rd_data = RegInit(0.U(64.W))
  val rd = RegInit(0.U(5.W))

  val canResp = RegInit(false.B)
  val acc_busy = RegInit(false.B)
  val configCompleted = RegInit(false.B)

  // config reg
  val RegionAttrBase = RegInit(0.U(MMUAddrWidth.W))
  val RegionAttrBiasedBase = RegInit(0.U(MMUAddrWidth.W))
  val RegionAttrShiftBy = RegInit(0.U(32.W))
  val HeapRegionBias = RegInit(0.U(32.W))
  val HeapRegionShiftBy = RegInit(0.U(32.W))
  val HeapRegionBiasedBase = RegInit(0.U(MMUAddrWidth.W))
  val HumongousReclaimCandidatesBoolBase = RegInit(0.U(MMUAddrWidth.W))
  val ParScanThreadStatePtr = RegInit(0.U(MMUAddrWidth.W))
  val TaskQueue_BottomAddr = RegInit(0.U(MMUAddrWidth.W))
  val TaskQueue_AgeTopAddr = RegInit(0.U(MMUAddrWidth.W))
  val TaskQueue_ElemsBase = RegInit(0.U(MMUAddrWidth.W))

  rd := io.cmd.bits.inst.rd    //下一拍一定会返回
  io.resp.bits.rd := rd
  io.resp.bits.data := rd_data
  io.resp.valid := canResp

  io.cmd.ready := !canResp
  // xd 表示 inst needs wirte regfile
  when(io.cmd.fire && io.cmd.bits.inst.xd === true.B){
    canResp := true.B
  }.elsewhen(io.resp.fire){
    canResp := false.B
  }

  // 使用 5B opcode
  // funct(6,0) === 0
  when(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 0.U){
    // config regionAttrBase and BiasedBase
    RegionAttrBase := io.cmd.bits.rs1
    RegionAttrBiasedBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 1.U){
    RegionAttrShiftBy := io.cmd.bits.rs1(63, 32)
    HeapRegionBias := io.cmd.bits.rs1(31, 0)
    HeapRegionShiftBy := io.cmd.bits.rs2(31, 0)
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 2.U){
    HeapRegionBiasedBase := io.cmd.bits.rs1
    HumongousReclaimCandidatesBoolBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 3.U){
    ParScanThreadStatePtr := io.cmd.bits.rs1
    TaskQueue_BottomAddr := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 4.U){
    TaskQueue_AgeTopAddr := io.cmd.bits.rs1
    TaskQueue_ElemsBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 5.U){
    rd_data := acc.io.ctrl2top.Done
  }

  when(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 4.U){
    configCompleted := true.B
  }.elsewhen(acc.io.ctrl2top.Valid && acc.io.ctrl2top.Ready){
    configCompleted := false.B
  }

  acc.io.ctrl2top.Valid := configCompleted
  acc.io.ctrl2top.RegionAttrBase := RegionAttrBase
  acc.io.ctrl2top.RegionAttrBiasedBase := RegionAttrBiasedBase
  acc.io.ctrl2top.RegionAttrShiftBy := RegionAttrShiftBy
  acc.io.ctrl2top.HeapRegionBias := HeapRegionBias
  acc.io.ctrl2top.HeapRegionShiftBy := HeapRegionShiftBy
  acc.io.ctrl2top.HeapRegionBiasedBase := HeapRegionBiasedBase
  acc.io.ctrl2top.HumongousReclaimCandidatesBoolBase := HumongousReclaimCandidatesBoolBase
  acc.io.ctrl2top.ParScanThreadStatePtr := ParScanThreadStatePtr
  acc.io.ctrl2top.TaskQueue_BottomAddr := TaskQueue_BottomAddr
  acc.io.ctrl2top.TaskQueue_AgeTopAddr := TaskQueue_AgeTopAddr
  acc.io.ctrl2top.TaskQueue_ElemsBase := TaskQueue_ElemsBase

  mem.io.mmu <> acc.io.mmu2llc
}





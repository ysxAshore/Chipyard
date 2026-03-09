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
  val id = WireInit(0.U(SourceMaxNumBitSize.W)) // 从最小的开始找, 一直覆盖 保留最大的空闲id

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

  io.mmu.Request.ready := tl_out.a.ready && !is_full
  when(io.mmu.Request.fire){
    if(DebugEnable){
      printf("[GCAcc2YGJK.node]sourceid: %x\n", io.mmu.ConherentRequsetSourceID.bits)
      printf("[GCAccYGJK.node.io.mmu.Request.bits] RequestType_isWrite %x, RequestPhysicalAddr %x, RequestData %x\n", io.mmu.Request.bits.RequestType_isWrite, io.mmu.Request.bits.RequestVirtualAddr, io.mmu.Request.bits.RequestData)
    }
  }

  // tilelink 执行内存访问操作有两个最基本的通道:A and D
  //    A: 传输对指定地址范围进行操作的请求
  //    D: 向原始请求者发送数据想要或者确认消息
  when(!is_full){
    when(tl_out.a.fire){
      busy(id) := true.B
    }
  }.otherwise{
    if(DebugEnable){
      printf("[GCAcc2Interface.node]sourceId queue is_full\n")
    }
  }

  // AccessAck 响应 Put 操作(写) AccessAckData 响应 Get 操作(读)
  // 本来是"read 需要等待上游是否可以确认接受 但是 write 写回的ack直接确认" 但是"这里write也需要返回给上级接收 所以不用直接置true"
  tl_out.d.ready := io.mmu.Response.ready
  //when(tl_out.d.bits.opcode === TLMessages.AccessAck && tl_out.d.valid && busy(tl_out.d.bits.source)){
  //  tl_out.d.ready := true.B
  //}

  when(tl_out.d.fire){
    busy(tl_out.d.bits.source) := false.B
    if(DebugEnable){
      printf("[GccAcc2YGJK.node]tl_out.d.fire: %x, tl_out.d.data: %x\n", tl_out.d.bits.source, tl_out.d.bits.data)
    }
  }

  tl_out.a.valid := io.mmu.Request.valid && !is_full
  tl_out.a.bits := Mux1H(Seq(
    (io.mmu.Request.bits.RequestType_isWrite === 0.U) -> edge.Get(id, io.mmu.Request.bits.RequestVirtualAddr, log2Ceil(MMUDataWidth / 8).U)._2, // id addr size
    (io.mmu.Request.bits.RequestType_isWrite === 1.U) -> edge.Put(id, io.mmu.Request.bits.RequestVirtualAddr, log2Ceil(MMUDataWidth / 8).U, io.mmu.Request.bits.RequestData, io.mmu.Request.bits.RequestWStrb)._2
  ))

  // read和write都需要返回Response
  io.mmu.Response.valid := tl_out.d.valid && (tl_out.d.bits.opcode === TLMessages.AccessAckData || tl_out.d.bits.opcode === TLMessages.AccessAck)
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

  // config reg
  val ChunkSize             = RegInit(0.U(32.W))
  val AgeThreshold          = RegInit(0.U(32.W))
  val HeapRegionBias        = RegInit(0.U(32.W))
  val RegionAttrShiftBy     = RegInit(0.U(32.W))
  val HeapRegionShiftBy     = RegInit(0.U(32.W))
  val LogOfHRGrainBytes     = RegInit(0.U(32.W))
  val StepperOffset         = RegInit(0.U(GCElementWidth.W))
  val YoungWordsBase        = RegInit(0.U(GCElementWidth.W))
  val RegionAttrBase        = RegInit(0.U(GCElementWidth.W))
  val PlabAllocatorPtr      = RegInit(0.U(GCElementWidth.W))
  val RegionAttrBiasedBase  = RegInit(0.U(GCElementWidth.W))
  val HeapRegionBiasedBase  = RegInit(0.U(GCElementWidth.W))
  val ParScanThreadStatePtr = RegInit(0.U(GCElementWidth.W))
  val TaskQueue_BottomAddr  = RegInit(0.U(GCElementWidth.W))
  val TaskQueue_ElemsBase   = RegInit(0.U(GCElementWidth.W))
  val HumongousReclaimCandidatesBoolBase = RegInit(0.U(GCElementWidth.W))
  val CardTablePtr          = RegInit(0.U(GCElementWidth.W))
  val G1h = RegInit(0.U(GCElementWidth.W))
  val IntArrayKlassObj = RegInit(0.U(GCElementWidth.W))
  val ObjectKlass = RegInit(0.U(GCElementWidth.W))
  val LockPtr = RegInit(0.U(GCElementWidth.W))
  val Thread = RegInit(0.U(GCElementWidth.W))
  val DummyRegion = RegInit(0.U(GCElementWidth.W))
  val NumaPtr = RegInit(0.U(GCElementWidth.W))
  val CompressedOopBase = RegInit(0.U(GCElementWidth.W))
  val CompressedKlassPointerBase = RegInit(0.U(GCElementWidth.W))
  val CompressedFlag = RegInit(0.U(32.W))

  val canResp = RegInit(false.B)
  val rd_data = RegInit(0.U(64.W))
  val rd = RegInit(0.U(5.W))

  io.resp.bits.rd := rd
  io.resp.bits.data := rd_data
  io.resp.valid := canResp

  io.cmd.ready := !canResp // 当存在响应未被CPU处理时 禁止接受新指令
  // xd 表示 inst needs wirte regfile
  when(io.cmd.fire){
    rd := io.cmd.bits.inst.rd
  }
  when(io.cmd.fire && io.cmd.bits.inst.xd === true.B){
    canResp := true.B
  }.elsewhen(io.resp.fire){
    canResp := false.B
  }

  // profile counter
  val acc_busy = RegInit(false.B)
  val count = RegInit(0.U(64.W))
  val memNum_r = RegInit(0.U(64.W))
  val memNum_w = RegInit(0.U(64.W))
  when(acc.io.ctrl2top.Valid && acc.io.ctrl2top.Ready){
    acc_busy := true.B
  }.elsewhen(acc.io.ctrl2top.Done){
    acc_busy := false.B
  }
  when(acc_busy){
    count := count + 1.U
  }
  when(acc.io.mmu2llc.Request.fire){
    when(acc.io.mmu2llc.Request.bits.RequestType_isWrite){
      memNum_w := memNum_w + 1.U
    }.otherwise{
      memNum_r := memNum_r + 1.U
    }
  }

  // 使用 5B opcode
  // funct(6,0) === 0
  when(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 0.U){
    ChunkSize:= io.cmd.bits.rs1(31, 0)
    AgeThreshold := io.cmd.bits.rs1(63, 32)
    HeapRegionBias := io.cmd.bits.rs2(31, 0)
    RegionAttrShiftBy := io.cmd.bits.rs2(63, 32)
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 1.U){
    HeapRegionShiftBy := io.cmd.bits.rs1(31, 0)
    LogOfHRGrainBytes := io.cmd.bits.rs1(63, 32)
    StepperOffset := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 2.U){
    YoungWordsBase := io.cmd.bits.rs1
    RegionAttrBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 3.U){
    PlabAllocatorPtr := io.cmd.bits.rs1
    RegionAttrBiasedBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 4.U) {
    HeapRegionBiasedBase := io.cmd.bits.rs1
    ParScanThreadStatePtr := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 5.U) {
    TaskQueue_BottomAddr := io.cmd.bits.rs1
    TaskQueue_ElemsBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 6.U) {
    HumongousReclaimCandidatesBoolBase := io.cmd.bits.rs1
    CardTablePtr := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 7.U) {
    G1h := io.cmd.bits.rs1
    IntArrayKlassObj := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 8.U) {
    ObjectKlass := io.cmd.bits.rs1
    LockPtr := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 9.U) {
    Thread := io.cmd.bits.rs1
    DummyRegion := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 10.U) {
    NumaPtr := io.cmd.bits.rs1
    CompressedOopBase := io.cmd.bits.rs2
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 11.U){
    CompressedKlassPointerBase := io.cmd.bits.rs1
    CompressedFlag := io.cmd.bits.rs2(31, 0)
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 16.U){
    rd_data := acc_busy
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 17.U){
    rd_data := count
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 18.U){
    rd_data := memNum_r;
  }.elsewhen(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 19.U){
    rd_data := memNum_w;
  }

  val configCompleted = RegInit(false.B)
  when(io.cmd.fire && io.cmd.bits.inst.opcode === "h5B".U && io.cmd.bits.inst.funct === 11.U){
    configCompleted := true.B
  }.elsewhen(acc.io.ctrl2top.Valid && acc.io.ctrl2top.Ready){
    configCompleted := false.B
  }

  acc.io.ctrl2top.Valid := configCompleted
  acc.io.ctrl2top.ChunkSize := ChunkSize
  acc.io.ctrl2top.CardTablePtr := CardTablePtr
  acc.io.ctrl2top.AgeThreshold := AgeThreshold
  acc.io.ctrl2top.StepperOffset := StepperOffset
  acc.io.ctrl2top.YoungWordsBase := YoungWordsBase
  acc.io.ctrl2top.RegionAttrBase := RegionAttrBase
  acc.io.ctrl2top.HeapRegionBias := HeapRegionBias
  acc.io.ctrl2top.PlabAllocatorPtr := PlabAllocatorPtr
  acc.io.ctrl2top.RegionAttrShiftBy := RegionAttrShiftBy
  acc.io.ctrl2top.HeapRegionShiftBy := HeapRegionShiftBy
  acc.io.ctrl2top.LogOfHRGrainBytes := LogOfHRGrainBytes
  acc.io.ctrl2top.RegionAttrBiasedBase := RegionAttrBiasedBase
  acc.io.ctrl2top.HeapRegionBiasedBase := HeapRegionBiasedBase
  acc.io.ctrl2top.ParScanThreadStatePtr := ParScanThreadStatePtr
  acc.io.ctrl2top.TaskQueue_BottomAddr := TaskQueue_BottomAddr
  acc.io.ctrl2top.TaskQueue_ElemsBase := TaskQueue_ElemsBase
  acc.io.ctrl2top.HumongousReclaimCandidatesBoolBase := HumongousReclaimCandidatesBoolBase
  acc.io.ctrl2top.G1h := G1h
  acc.io.ctrl2top.IntArrayKlassObj := IntArrayKlassObj
  acc.io.ctrl2top.ObjectKlass := ObjectKlass
  acc.io.ctrl2top.LockPtr := LockPtr
  acc.io.ctrl2top.Thread := Thread
  acc.io.ctrl2top.DummyRegion := DummyRegion
  acc.io.ctrl2top.NumaPtr := NumaPtr
  acc.io.ctrl2top.CompressedOopBase := CompressedOopBase
  acc.io.ctrl2top.CompressedKlassPointerBase := CompressedKlassPointerBase
  acc.io.ctrl2top.CompressedFlag := CompressedFlag

  mem.io.mmu <> acc.io.mmu2llc
}
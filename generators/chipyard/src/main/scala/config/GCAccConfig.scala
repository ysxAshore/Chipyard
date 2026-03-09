package chipyard

import boom.v3.common.{BoomCoreParams, BoomTileAttachParams, WithNSmallBooms}
import chipyard.config.{AbstractConfig, WithSV48, WithSystemBusWidth}
import freechips.rocketchip.subsystem.{ExtMem, InSubsystem, TilesLocated}
import org.chipsalliance.cde.config.Config

class GCAccSmallBoomConfig extends Config(
  new WithSV48 ++
  new spinal_gc.WithGCAcc(Seq(0)) ++
  new WithSystemBusWidth(256) ++
  new WithNSmallBooms(1) ++
  new Config((site, here, up) => {
    case ExtMem => up(ExtMem, site).map(m => m.copy(
      master = m.master.copy(
        base = BigInt("FE00000000", 16),
        size = BigInt(1) << 33
      )
    ))
  }) ++
  new AbstractConfig)
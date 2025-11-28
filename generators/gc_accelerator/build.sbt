version := "1.0"
name := "gc_accelerator"
scalaVersion := "2.13.10"
Compile / packageBin / mappings ~= { mappings =>
  mappings.filter { case (_, path) =>
    !path.startsWith("gc_tests/")
  }
}
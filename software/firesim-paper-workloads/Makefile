WORKLOAD_DIR=$(abspath .)

# TODO: re-enable memcached-thread-imbalance (disabled since sysroot needs to be re-built instead of cached in newer version of FireMarshal)
.PHONY: allpaper
allpaper: simperf-test-latency simperf-test-scale bw-test-two-instances ping-latency

FIREMARSHAL_WORKLOAD_NAME=br-base
BASE_IMAGE:=$(WORKLOAD_DIR)/../firemarshal/images/firechip/$(FIREMARSHAL_WORKLOAD_NAME)/$(FIREMARSHAL_WORKLOAD_NAME).img
BASE_LINUX:=$(WORKLOAD_DIR)/../firemarshal/images/firechip/$(FIREMARSHAL_WORKLOAD_NAME)/$(FIREMARSHAL_WORKLOAD_NAME)-bin
$(BASE_IMAGE) $(BASE_LINUX) &:
	marshal -v build $(FIREMARSHAL_WORKLOAD_NAME).json

GAP_DIR=runscripts/gapbs-scripts
$(GAP_DIR)/overlay/$(input):
	cd $(GAP_DIR) && ./gen_run_scripts.sh --binaries --input $(input)

#Default to test input size for GAPBS
.PHONY: gapbs
gapbs: input = graph500
gapbs: gapbs.json $(GAP_DIR)/overlay/$(input) $(BASE_IMAGE) $(BASE_LINUX)
	mkdir -p $@
	cp $(BASE_LINUX) $@/bbl-vmlinux
	python3 gen-benchmark-rootfs.py -w $< -r -b $(BASE_IMAGE) \
		-s $(GAP_DIR)/overlay/$(input) \

.PHONY: memcached-thread-imbalance
memcached-thread-imbalance:
	mkdir -p $@
	cd $@ && git submodule update --init mutilate-loadgen-riscv-release
	cd $@/mutilate-loadgen-riscv-release && ./build.sh
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE) -s $@/mutilate-loadgen-riscv-release/overlay

.PHONY: bw-test-two-instances
bw-test-two-instances: bw-test-two-instances.json $(BASE_LINUX) $(BASE_IMAGE)
	cd $(WORKLOAD_DIR)/network-benchmarks && python3 build-bw-test.py -n 8
	cp $(WORKLOAD_DIR)/network-benchmarks/testbuild/*.riscv $@

.PHONY: bw-test-one-instance
bw-test-one-instance: bw-test-one-instance.json $(BASE_LINUX) $(BASE_IMAGE)
	cd $(WORKLOAD_DIR)/network-benchmarks && python3 build-bw-test.py -n 4
	cp $(WORKLOAD_DIR)/network-benchmarks/testbuild/*.riscv $@

.PHONY: ping-latency
ping-latency: $(BASE_LINUX) $(BASE_IMAGE)
	mkdir -p $@
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE) -s $@/overlay

.PHONY: simperf-test
simperf-test:  $(BASE_LINUX) $(BASE_IMAGE)
	mkdir -p $@
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE) -s $@/overlay

.PHONY: simperf-test-scale simperf-test-latency flash-stress
simperf-test-scale: simperf-test
simperf-test-latency: simperf-test
flash-stress: simperf-test-latency

.PHONY: linux-poweroff
linux-poweroff: $(BASE_LINUX) $(BASE_IMAGE)
	mkdir -p $@/overlay
	cd $(WORKLOAD_DIR)/check-rtc && make print-mcycle-linux
	cp $(WORKLOAD_DIR)/check-rtc/print-mcycle-linux $@/overlay/
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE) -s $@/overlay

.PHONY: iperf3
iperf3: iperf3.json $(BASE_LINUX) $(BASE_IMAGE)
	mkdir -p $@
	cd $@ && ln -sf ../$(BASE_LINUX) bbl-vmlinux
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE)

.PHONY: check-rtc
check-rtc:
	cd $(WORKLOAD_DIR)/check-rtc && make check-rtc

.PHONY: check-rtc-linux
check-rtc-linux:
	mkdir -p $@/overlay
	cd $(WORKLOAD_DIR)/check-rtc && make check-rtc-linux
	cp $(WORKLOAD_DIR)/check-rtc/check-rtc-linux $@/overlay
	cd $@ && ln -sf ../$(BASE_LINUX) bbl-vmlinux
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE) -s $@/overlay

.PHONY: checksum-test
checksum-test:
	cd ../../target-design/chipyard/tests && make checksum.riscv

.PHONY: ccbench-cache-sweep
ccbench-cache-sweep:
	cd ccbench-cache-sweep/ccbench/caches && make ARCH=riscv
	python3 gen-benchmark-rootfs.py -w $@.json -r -b $(BASE_IMAGE) -s $@/

.PHONY: fc-test
fc-test:
	cd $(WORKLOAD_DIR)/network-benchmarks/fc-test && make
	ln -sf ../$(WORKLOAD_DIR)/network-benchmarks/fc-test/fc-client.riscv $@/fc-client.riscv
	ln -sf ../$(WORKLOAD_DIR)/network-benchmarks/fc-test/fc-server.riscv $@/fc-server.riscv

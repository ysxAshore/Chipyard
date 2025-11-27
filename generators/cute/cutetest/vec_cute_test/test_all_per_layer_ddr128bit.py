import os
from multiprocessing import Pool, current_process, Manager
import subprocess

def run_command(x, progress, total, lock):
    process_name = current_process().name
    print(f"{process_name}: 开始执行 vec_ops_conv_{x}.riscv")

    # 创建一个临时的bash脚本来执行所有命令
    script_content = f"""
    source /root/chipyard/env.sh
    /root/chipyard/scripts/smartelf2hex.sh /root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_cute_test/vec_ops_conv_{x}.riscv > /root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig-128bitddr-512bit-l2/vec_ops_conv_{x}.loadmem_hex
    (set -o pipefail && /root/chipyard/sims/verilator/simulator-chipyard-CUTETestConfig +permissive +dramsim \\
    +dramsim_ini_dir=/root/chipyard/generators/testchipip/src/main/resources/dramsim2_ini +max-cycles=1000000000 \\
    +loadmem=/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig-128bitddr-512bit-l2/vec_ops_conv_{x}.loadmem_hex \\
    +loadmem_addr=80000000 +testfile=/root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_cute_test/vec_ops_conv_{x}.riscv \\
    +whisper_path=/root/.cache/bazel/_bazel_root/b724f9849c2f0e03f9d26146ce9ac229/execroot/_main/bazel-out/k8-fastbuild/bin/external/whisper/whisper \\
    +whisper_json_path=/root/chipyard/sims/cosim/bridge/whisper/config/boom.json +bootcode=/root/chipyard/sims/cosim/bootrom/bootrom \\
    +verbose +permissive-off /root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_cute_test/vec_ops_conv_{x}.riscv \\
    </dev/null 2> >(spike-dasm > /root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig-128bitddr-512bit-l2/vec_ops_conv_{x}.out) \\
    | tee /root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig-128bitddr-512bit-l2/vec_ops_conv_{x}.log)
    """
    


    script_path = f"/tmp/run_vec_ops_conv_{x}.sh"
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    # 给予执行权限
    os.chmod(script_path, 0o755)

    # 执行bash脚本
    subprocess.run(script_path, shell=True, executable="/bin/bash")

    print(f"{process_name}: 完成执行 vec_ops_conv_{x}.riscv")

    # 更新进度
    with lock:
        progress.value += 1
        percent_complete = (progress.value / total) * 100
        print(f"总进度: {percent_complete:.2f}%")

if __name__ == "__main__":
    total_tasks = 53 - 2 + 1  # 总任务数
    manager = Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    lock = manager.Lock()  # 锁对象

    # 创建一个包含24个进程的进程池
    with Pool(24) as p:
        # 并行执行vec_ops_conv_2.riscv到vec_ops_conv_53.riscv
        p.starmap(run_command, [(x, progress, total_tasks, lock) for x in range(2, 54)])
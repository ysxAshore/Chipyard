import os
from multiprocessing import Pool, current_process, Manager
import subprocess

def run_command(x, progress, total, lock):
    process_name = current_process().name
    print(f"{process_name}: 开始执行 {x}.riscv")
# /root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_test_t/vec_ops_kernel_shift_scale.riscv
    # 创建一个临时的bash脚本来执行所有命令
    script_content = f"""
    source /root/chipyard/env.sh
    /root/chipyard/scripts/smartelf2hex.sh /root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_test_t/{x}.riscv > /root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/{x}.loadmem_hex
    (set -o pipefail && /root/chipyard/sims/verilator/simulator-chipyard-CUTETestConfig128bitdram512bitL2Widen3issueBoom +permissive +dramsim \\
    +dramsim_ini_dir=/root/chipyard/generators/testchipip/src/main/resources/dramsim2_ini +max-cycles=1000000000 \\
    +loadmem=/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/{x}.loadmem_hex \\
    +loadmem_addr=80000000 +testfile=/root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_test_t/{x}.riscv \\
    +whisper_path=/root/.cache/bazel/_bazel_root/b724f9849c2f0e03f9d26146ce9ac229/execroot/_main/bazel-out/k8-fastbuild/bin/external/whisper/whisper \\
    +whisper_json_path=/root/chipyard/sims/cosim/bridge/whisper/config/boom.json +bootcode=/root/chipyard/sims/cosim/bootrom/bootrom \\
    +verbose +permissive-off /root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_test_t/{x}.riscv \\
    </dev/null 2> >(spike-dasm > /root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/{x}.out) \\
    | tee /root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig128bitdram512bitL2Widen3issueBoom/{x}.log)
    """

    script_path = f"/tmp/run_{x}.sh"
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    # 给予执行权限
    os.chmod(script_path, 0o755)

    # 执行bash脚本
    subprocess.run(script_path, shell=True, executable="/bin/bash")

    print(f"{process_name}: 完成执行 {x}.riscv")

    # 更新进度
    with lock:
        progress.value += 1
        percent_complete = (progress.value / total) * 100
        print(f"总进度: {percent_complete:.2f}%")

if __name__ == "__main__":
    #查询当前路径下的所有.riscv文件,并将其作为任务列表
    
    total_tasks = 0
    file_list = []
    for file in os.listdir('/root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/vec_test_t/'):
        if file.endswith('.riscv'):
            total_tasks += 1
            file_list.append(file[:-6])
            print(file_list)
            
    #一口气执行所有任务
    manager = Manager()
    progress = manager.Value('i', 0)  # 进度计数器
    lock = manager.Lock()  # 锁对象
    
    # 创建一个包含24个进程的进程池
    with Pool(24) as p:
        # 并行执行conv_params_2.riscv到conv_params_53.riscv
        p.starmap(run_command, [(x, progress, total_tasks, lock) for x in file_list])
    
# 将指定长度二进制转换为有符号整数
def hex2sint(num, length):
    # 将num用特定长度length二进制表示出来
    binary = bin(num & int("1"*length, 2))
    # 将binary填充到length长度
    binary = binary[2:].zfill(length)
    # 如果是负数
    if binary[0] == "1":
        return int(binary, 2) - 2**length
    else:
        return int(binary, 2)
    
# 这个golden检查用 单个 macro操作完成的resnet50的卷积操作的CML写回的地址和数据的正确性
# 通过计算预期的CML写回的地址来从golden数组中找到预期的数据，这在之后的resnet50的完整操作中将不起作用 
# 因为后续的resnet50将进行软件分块，部分的CML写回将只是卷积的中间结果而不是最终结果

# 这里通过读取.h里提前算好的golden来得到卷积的预期值
# 也可以在这里通过读出的input、weight和bias模拟卷积算出来现场得到matrix_golden，这样在后面做resnet50的完整操作没有卷积的golden时检查卷积操作的中间结果是否正确
for test_index in range(2, 3):
    # 输入文件名
    input_file = f"/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig/conv_params_{test_index}.out"  # 要筛选的trace文件
    output_file = "/root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/test_gen/resnet50/CML_Store_trace.out"  # 保存筛选结果的文件

    # 过滤出特定部件的trace
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith("[CMemoryLoader_Store<"):
                outfile.write(line)

    print(f"筛选完成，结果已保存到 {output_file}")

    app_m = 0
    app_k = 0
    app_n = 0
    martix_A = []
    martix_B = []
    martix_D = []
    martix_golden_c = []
    golden_base = 0
    base_get = False


    # 读取.h文件的所有内容
    with open(f"./conv_{test_index}.h", "r") as f:
        # 读出特定的参数
        content = f.read()
        app_m = int(content.split("#define APPLICATION_M ")[1].split("\n")[0])
        app_k = int(content.split("#define APPLICATION_K ")[1].split("\n")[0])
        app_n = int(content.split("#define APPLICATION_N ")[1].split("\n")[0])
        # data = content.split(f"static char input[49][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        # data = content.split("static char a[113][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        # martix_A = eval(data.replace("{", "[").replace("}", "]"))
        # print(martix_A)
        
        # data = content.split("static char weight[128][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        # martix_B = eval(data.replace("{", "[").replace("}", "]"))
        # print(martix_B)
        
        # data = content.split("static int bias[128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        # data = content.split("static int d[113][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        # martix_D = eval(data.replace("{", "[").replace("}", "]"))
        
        # 找到static int gloden_c[113][128] __attribute__((aligned(256))) = 后 ; 之前的内容
        data = content.split(f"static int gloden_output_with_scale[APPLICATION_M*APPLICATION_N] __attribute__((aligned(64))) = ")[1].split(";")[0]
        # data = content.split("static int gloden_c[113][128] __attribute__((aligned(256))) = ")[1].split(";")[0]
        martix_golden_c = eval(data.replace("{", "[").replace("}", "]"))

    input_file = "./CML_Store_trace.out"

    store_request = []
    request_index = 0

    with open(input_file, "r") as infile:
        taskid = 0
        Matrix_M = 4
        scp_n = 64
        scp_m = 64
        MLEN = 32
        c_scp_entry_size = 16
        d_datatype = 4
        per_store_reduce_dim_iter = MLEN // d_datatype
        max_subreduce = c_scp_entry_size * Matrix_M // d_datatype
        max_submajor = Matrix_M
        
        app_stride = app_n * d_datatype
        currentstore_blocktensor_major_dim_iter = 0
        currentstore_blocktensor_reduce_dim_iter = 0
        
        tensor_block_baseaddr = 0
        # 遍历文件每一行
        for line in infile:
            if line.find("Store D Tensor Start") != -1:
                # app_m不是scp_m整倍数时，末尾的scp_m不再是硬件设置的最大scp_m
                if taskid >= app_m // scp_m * (app_n // scp_n):
                    scp_m = app_m % scp_m
                
                # 每一个task都会进行一系列的写回操作，在解析随后的写回trace前提前将写回操作的地址偏移量序列算出来
                store_request.clear()
                request_index = 0
                max_load_scp_time = scp_n * scp_m * d_datatype // MLEN
                currentstore_blocktensor_submajor_dim_iter = 0
                currentstore_blocktensor_major_dim_iter = 0
                currentstore_blocktensor_subreduce_dim_iter = 0
                currentstore_blocktensor_reduce_dim_iter = 0
                
                for total_store_size in range(max_load_scp_time):
                    store_request.append((currentstore_blocktensor_major_dim_iter + currentstore_blocktensor_submajor_dim_iter) * app_stride 
                                        + (currentstore_blocktensor_reduce_dim_iter + currentstore_blocktensor_subreduce_dim_iter) * d_datatype)
                    currentstore_blocktensor_subreduce_dim_iter += per_store_reduce_dim_iter
                    if currentstore_blocktensor_subreduce_dim_iter >= max_subreduce:
                        currentstore_blocktensor_subreduce_dim_iter = 0
                        currentstore_blocktensor_submajor_dim_iter += 1
                        # CScratchpad将每Matrix_M个m为单位重排序，若scp_m不是Matrix_M的整数倍，则末尾的CScratchpad取数不再是连续的
                        if currentstore_blocktensor_submajor_dim_iter >= max_submajor or (currentstore_blocktensor_major_dim_iter + currentstore_blocktensor_submajor_dim_iter) >= scp_m:
                            currentstore_blocktensor_submajor_dim_iter = 0
                            currentstore_blocktensor_reduce_dim_iter += max_subreduce
                            if currentstore_blocktensor_reduce_dim_iter >= scp_n:
                                currentstore_blocktensor_reduce_dim_iter = 0
                                currentstore_blocktensor_major_dim_iter += Matrix_M
                                
                taskid += 1
                
                
            if line.find("WriteRequest: RequestVirtualAddr=") != -1:
                addr_trace = int(line.split("RequestVirtualAddr=")[1].split(",")[0], 16)
                if (request_index == 0):
                    tensor_block_baseaddr = addr_trace
                    # 第一次任务的tensor_block_baseaddr为输出数组的首地址，记录下来并通过这个地址计算地址偏移量来得到写回地址对应的数组下标
                    if (base_get == False):
                        base_get = True
                        golden_base = addr_trace
                
                addr_golden = tensor_block_baseaddr + store_request[request_index]
                if (addr_trace != addr_golden):
                    print(f"request_index:{request_index},addr_trace:{hex(addr_trace)},addr_golden:{hex(addr_golden)}")
                    exit(0)
                    
                data_trace_hex = line.split("RequestData:")[1].split("\n")[0]
                # print(data_trace_hex)
                # 将trace的十六进制数据流转换成整数数组来与golden进行对比
                data_trace_hex = [data_trace_hex[i:i+2*d_datatype] for i in range(0, len(data_trace_hex), 2*d_datatype)][::-1]
                data_trace = [hex2sint(int(x, 16), 8 * d_datatype) for x in data_trace_hex]
                
                offset = int(addr_golden - golden_base) // d_datatype
                # print(offset)
                data_golden = martix_golden_c[offset // app_n][offset % app_n : offset % app_n + Matrix_M // d_datatype]
                if (data_golden != data_trace):
                    print(f"request_index{request_index},addr{hex(addr_golden)},data_trace{data_trace},data_golden{data_golden}")
                    exit(0)
            
                request_index += 1
                

            
    print("Down!")
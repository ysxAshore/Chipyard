import re

# 读取resnet50_layer_param.txt文件
with open('resnet50_layer_param.txt', 'r') as file:
    lines = file.readlines()

# 定义正则表达式来解析参数
pattern = re.compile(r'ih iw ic = (\d+) (\d+) (\d+),\s+oh ow oc = (\d+) (\d+) (\d+),\s+kh kw ic oc = (\d+) (\d+) (\d+) (\d+),\s+stride = (\d+)')

# 遍历每一行，计算L2的Load和L2的Store次数
for index, line in enumerate(lines):
    match = pattern.search(line)
    if match:
        ih, iw, ic, oh, ow, oc, kh, kw, ic2, oc2, stride = map(int, match.groups())

        # 计算L2的Load次数
        k1_l2_A_load = ow * oh * (ic//32) * (oc //64)
        #                   VVVVVVVVVVVVVVkernelVVVVVVVVVVVVVVVVV    VTile_N   VTil_K   V每个tile的reducedim的贡献
        k3_s1_l2_A_load = (ow*ow + (ow-1)*(ow-1)*4 + (ow-1)*ow*4) * (oc//64) * (ic//64) * 2
        k3_s2_l2_A_load = (ow*ow*4 + (ow-1)*(ow-1)*1 + (ow-1)*ow*4) * (oc//64) * (ic//64) * 2
        
        l2_B_load = ((ow*oh+63)//64 * oc//64 * ic//64) * kh * kw * 64 * 2
        
        l2_C_load = (oc*4//32)*((ow*oh+63)//64)
        
        l2_D_store = (ow*oh*oc*4//32)
        
        l2_stores = l2_D_store
        
        l2_A_load = k1_l2_A_load
        if kh == 3 and stride == 1:
            l2_A_load = k3_s1_l2_A_load
        elif kh == 3 and stride == 2:
            l2_A_load = k3_s2_l2_A_load
        
        l2_loads = l2_A_load + l2_B_load + l2_C_load
        
        print(f"Layer({index+2}): ih={ih}, iw={iw}, ic={ic}, oh={oh}, ow={ow}, oc={oc}, kh={kh}, kw={kw}, stride={stride}")
        # print(f"L2 Loads: {l2_loads}")
        # print(f"L2 A Loads: {l2_A_load}")
        # print(f"L2 B Loads: {l2_B_load}")
        # print(f"L2 C Loads: {l2_C_load}")
        # print(f"L2 Stores: {l2_stores}")
        
        #获取/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig/conv_params_x.log
        #获取L2的Load和Store次数
        #acc read req: 58368 格式如图所示
        #acc write req: 25088 格式如图所示
        
        ## 读取conv_params_x.log文件
        log_file = f"/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig/conv_params_{index+2}.log"
        with open(log_file, "r") as file:
            lines = file.readlines()
        
        # 定义正则表达式来解析参数
        pattern_read = re.compile(r'acc read req: (\d+)')
        pattern_write = re.compile(r'acc write req: (\d+)')
        read_req = 0
        write_req = 0
        # 遍历每一行，计算L2的Load和L2的Store次数
        for line in lines:
            match_read = pattern_read.search(line)
            match_write = pattern_write.search(line)
            if match_read:
                read_req = int(match_read.groups()[0])
            if match_write:
                write_req = int(match_write.groups()[0])
        
        if read_req != l2_loads or write_req != l2_stores:
            print(f"Error: L2 Loads mismatch! Expected {l2_loads}, got {read_req}")
            print(f"Error: L2 Stores mismatch! Expected {l2_stores}, got {write_req}")
            
        
                
        
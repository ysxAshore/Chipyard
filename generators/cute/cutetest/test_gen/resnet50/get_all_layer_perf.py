# 代码
import os
import re
# conv_params_49.log


def get_all_test_cycle():
    ##处理layer_param.txt文件。来获取每一层的计算量,放在一个字典里，key是conv_x,value是ops
    layer_param_dict = {}
    with open('resnet50_layer_param.txt', 'r') as file:
        #CONV_PARAMS(2) = {      ih iw ic = 56 56 64,    oh ow oc = 56 56 64,    kh kw ic oc = 1 1 64 64,        stride = 1, };  1(3136,64,64)1,padding=0,3136%64==0     Input_size:0.19MB,weight_size:0.00MB,output_size:0.77MB,Mops:12.85MMAC
        #读取每一行，处理得到MMAC，并转换为ops即可
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            #获取数字x
            num = re.findall(r"\d+", line)
            num = num[0]
            #print(num)
            #获取ops
            ops = re.findall(r"Mops:(\d+\.\d+)MMAC", line)
            ops = ops[0]
            #print(ops)
            #ops转换为浮点数然后乘以2
            layer_param_dict["conv_"+num] = float(ops) * 2 * 1000 * 1000
    print(layer_param_dict)
    with open("resnet50_layer_param.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # print(line)
            #获取数字x
            num = re.findall(r"\d+", line)
            num = num[0]
            print("conv_"+num)
            logpath = "/root/chipyard/sims/verilator/output/chipyard.TestHarness.CUTETestConfig/conv_params_49.log"
            logpath = logpath.replace("49", num)
            with open(logpath, "r") as f:
                content = f.readlines()
                totalcycle_line = "Cycles: "
                for logline in content:
                    if totalcycle_line in logline:
                        totalcycle = re.findall(r"Cycles: (\d+)", logline)
                        print("totalcycle = " + totalcycle[0])
                        print("op nums = " + str(layer_param_dict["conv_"+num]))
                        print("op per cycle = " + str(layer_param_dict["conv_"+num]/int(totalcycle[0])))
                        print()
                        break
                    
get_all_test_cycle()


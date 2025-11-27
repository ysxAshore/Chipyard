# 读取resnet50_conv_task.txt，标明了所有的测试用例的头文件，现在用resnet50_cpu_int8_CUTE_chipyard_run.c这个文件作为模板，生成.c文件
# .txt文件每一行都是一个头文件，如
# conv_2.h
# conv_3.h
# conv_4.h
# conv_5.h
# conv_6.h

# 我们要对应生成.c文件，如conv_2.c

# 其中需要修改resnet50_cpu_int8_CUTE_chipyard_run.c这个模板文件的内容如下
# tiled_conv_CUTE_auto(conv_2_params, input, weights, bias, output,ACT_TYPE);这一行的conv_2_params改成conv_x_params
# printf("conv_2 cycles: %lu \n", end - start); 这一行的conv_2改成conv_x
# #include "conv_2.h" 这一行的conv_2改成conv_x

# 代码
import os
import re

def get_all_test():
    with open("resnet50_conv_task.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            #获取数字x
            num = re.findall(r"\d+", line)
            num = num[0]
            print(num)
            with open("resnet50_cpu_int8_CUTE_chipyard_run_pure.c", "r") as f:
                content = f.read()
                #完整的替换一行，以免替换到其他地方
                content = content.replace("tiled_conv_CUTE_auto(conv_2_params, input, weights, bias, output,ACT_TYPE);", "tiled_conv_CUTE_auto(conv_" + num + "_params, input, weights, bias, output,ACT_TYPE);")
                content = content.replace("printf(\"conv_2 cycles: %lu \\n\", end - start);", "printf(\"conv_" + num + " cycles: %lu \\n\", end - start);")
                content = content.replace("#include \"conv_2.h\"", "#include \"" + line + "\"")
                with open("vec_ops_conv_" + num + ".c", "w") as f:
                    f.write(content)
                    
get_all_test()

# 运行后会生成conv_2.c, conv_3.c, conv_4.c, conv_5.c, conv_6.c这几个文件

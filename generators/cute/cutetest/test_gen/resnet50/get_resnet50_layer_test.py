import os
import shutil
import subprocess
import re

# 读取resnet50_layer_param.txt文件
with open('resnet50_layer_param.txt', 'r') as file:
    lines = file.readlines()

# 定义正则表达式来解析参数
pattern = re.compile(r'ih iw ic = (\d+) (\d+) (\d+),\s+oh ow oc = (\d+) (\d+) (\d+),\s+kh kw ic oc = (\d+) (\d+) (\d+) (\d+),\s+stride = (\d+)')

# 遍历每一行，生成对应的.h和.c文件
for index, line in enumerate(lines):
    # 使用正则表达式解析参数
    index = index +2
    match = pattern.search(line)
    if match:
        ih, iw, ic, oh, ow, oc, kh, kw, ic2, oc2, stride = map(int, match.groups())
        n = 1  # n永远等于1
        kernel_size = kh
        conv_stride = stride

        # 调用生成卷积参数的程序
        subprocess.run([
            "/root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/test_gen/resnet50/get_conv_test_cmd",
            str(n), str(ih), str(iw), str(ic), str(kernel_size), str(conv_stride), str(oh), str(ow), str(oc)
        ])

        # 重命名生成的conv_value.h文件
        h_filename = f"conv_params_{index}.h"
        os.rename("conv_value.h", h_filename)

        # 复制cutehello.c文件并重命名
        c_filename = f"conv_params_{index}.c"
        shutil.copy("/root/chipyard/generators/boom/src/main/resources/cutetest/cute_test_with_vec/test_gen/resnet50/cutehello.c", c_filename)

        # 修改.c文件中的include引用
        with open(c_filename, "r") as file:
            content = file.read()

        content = content.replace('#include "conv_params_2.h"', f'#include "{h_filename}"')

        with open(c_filename, "w") as file:
            file.write(content)

print("操作完成")
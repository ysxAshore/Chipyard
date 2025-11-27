# CUTE-Test ReadMe

本仓库是用于存储CUTE使用的神经网络推理的模型的仓库，由**gemmini**执行的推理代码移植而来。目前移植了resnet50和transfomer。

---

## [2024年12月30日]

### /imagenet[add !]
##### /imagenet/Makefile 
执行make指令，生成resnet50_dataflow，resnet50_CUTE_Gloden_int8，get_transpose，shift_int8，resnet50_gloden
- resnet50_cute_dataflow，打印resnet50的卷积任务的数据规模
- resnet50_cpu_int8_CUTE_Gloden，经过修改的CUTE可执行的int8纯整型量化的resnet50
- get_transpose，得到resnet50的权重矩阵的转置的函数
- shift_int8，通过纯整型量化优化的cpu执行的resnet50，删去了大部分的浮点操作
- resnet50_gloden，最初始的gemmini中摘取的，可由cpu运行的resnet50的
- get_layer_1_out，获取第一层输出
- get_all_layer_out，获取每一层卷积执行需要的输入输出和golden_result，输出为conv_x.h
##### /imagenet/conv_*.h
由gemmini中的纯整型量化的resnet50的模型中提取的，每一层卷积执行需要的输入输出和golden_result


### /test_gen[new !]
##### /test_gen/resnet50/Makefile
执行make指令，生成conv_params_x.riscv,该文件可由chipyard生成的CUTE的verilator模拟器执行
##### /test_gen/resnet50/conv_params_x.h
包含CUTE可识别的resnet50某层的参数和随机输入输出，可用于验证正确性
##### /test_gen/resnet50/conv_params_x.c
包含驱动CUTE执行resnet50某层的代码
##### /test_gen/resnet50/conv_params_x.riscv
可被，由chipyard生成的CUTE的verilator模拟器，执行的elf文件
##### /test_gen/resnet50/cuteMarcoinstHelper.h
包装了CUTE的各种宏指令功能的Helper函数
##### /test_gen/resnet50/get_conv_test_cmd.c
接受命令行参数，生成某个指定卷积层参数和随机输入还有GoldenResult的c文件
#####  /test_gen/resnet50/get_resnet50_layer_test.py
[*提前使用gcc get_conv_test_cmd.c -o get_conv_test_cmd，然后再执行本python文件*]
接受resnet50_layer_param.txt参数，调用get_conv_test_cmd，生成随机生成的resnet50的参数层
#####  /test_gen/resnet50/test_per_layer.py
[*提前使用python get_resnet50_layer_test.py，并执行make，然后再执行本python文件*]
并行的使用24个线程测试resnet50的每一层，得到每一层卷积的执行效果
#####  /test_gen/resnet50/test_per_error_layer.py
[*提前使用python get_resnet50_layer_test.py，并执行make，然后再执行本python文件*]
并行的使用24个线程测试resnet50的指定的某些层，得到句阿吉的执行效果
#####  /test_gen/resnet50/get_resnet50_layer_Load_Store_Checker.py
检查每个由test_per_layer.py或者test_per_error_layer.py执行的卷积层的结果，目前验证了Load_Store次数的验证




### /transformer[Waiting for update~]
transformer.................

### /vec_cute_test[new !]
##### /vec_cute_test/Makefile
执行make指令，可使用conv_x.h和vec_ops_conv_x.c生成vec_ops_conv_x.riscv，可在有向量指令支持的CUTEv2上的verilator上执行的elf
##### /vec_cute_test/vec_ops_conv_*.riscv
可被，由chipyard生成的CUTE的verilator模拟器，执行的elf文件
##### /vec_cute_test/conv_*.h
用gemmini的纯整数量化的resnet50中提取的每一层的参数和每一层的输入输出，由get_all_layer_out生成
##### /vec_cute_test/cuteMarcoinstHelper.h
包装了CUTE的各种宏指令功能的Helper函数
##### /vec_cute_test/resnet50_cpu_int8_CUTE_chipyard_run_pure.c
用于生成每一层CUTE-Vec层融合算子的模板代码，删去了耗时的printf
##### /vec_cute_test/resnet50_cpu_int8_CUTE_chipyard_run.c
用于生成每一层CUTE-Vec层融合算子的模板代码，保留了printf可用于debug
##### /vec_cute_test/get_all_test.py
生成每一层CUTE-Vec层融合算子的脚本，用于生成vec_ops_conv_x.c


---

## [2024年11月15日]

本仓库是用于存储CUTE使用的神经网络推理的模型的仓库，由gemmini执行的仓库移植而来。目前移植了resnet50和transfomer。

到指定目录下进行make即可。

目前正在移植中，目标实现CUTE执行resnet50的gloden trace和软件模拟。

目前两个硬件加速指令，卷积加速指令和矩阵乘加速指令。
卷积加速指令，要求的参数为：
CUTE_CONV_KERNEL_MarcoTask
(void *A,void *B,void *C,void *D,
int Application_M,int Application_N,int Application_K,
int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,
uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,
int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)

ABCD分别为矩阵A、矩阵B和结果矩阵C，偏置矩阵D的起始地址。要求所有矩阵都是Reduce_DIM_FIRST的
Application_M,Application_N,Application_K代表这条宏指令要执行的MNK的长度
conv_stride是卷积的stride步长
kernel_size是卷积核的大小
kernel_stride是每一个index的卷积核的大小，我们要求卷积核的数据排布是(kh,kw,oc,ic)
stride_A、stride_B,stride_C,stride_D代表各个矩阵Reduce_DIM的长度(多少byte)
transpose_result表示结果是否需要进行转置
conv_oh_index,conv_ow_index代表当前处理的矩阵A的起始地址，落在卷积任务input的哪个index上
conv_oh_max,conv_ow_max与index配合，可以完成padding、stride等操作的加速
void * VectorOp,int VectorInst_Length代表了要融合的向量任务的具体指令和指令块长度。

矩阵乘就是IH=1，IW=M，IC=K，OC=N，KH=1，KW=1，STRIDE=1的卷积
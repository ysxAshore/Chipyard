import sys
import re

def process_file(input_path, output_path):
    # 读取整个文件内容以查找模块声明部分
    with open(input_path, 'r') as f:
        content = f.readlines()

    # 定位模块声明起始行和结束行
    start_line = -1
    end_line = -1
    for i, line in enumerate(content):
        if line.strip().startswith('module ChipTop('):
            start_line = i
        if start_line != -1 and end_line == -1 and ');' in line:
            end_line = i
            break

    if start_line == -1 or end_line == -1:
        raise ValueError("Module declaration not found.")

    # 提取模块声明部分的所有行
    module_lines = content[start_line:end_line + 1]

    # 提取目标字符串
    pattern = re.compile(r'^\s*(input|output)\s+(?:\[.*?\]\s+)*(\w+)\b')
    original_strings = []
    for line in module_lines:
        matches = pattern.findall(line)
        for match in matches:
            original_strings.append(match[1])

    # 生成替换映射表，并按键的长度降序排序
    mapping = {}
    for s in original_strings:
        new_s = s.replace('_bits_', '_')
        new_s = new_s.replace('_ar_', '_ar')
        new_s = new_s.replace('_aw_', '_aw')
        new_s = new_s.replace('_r_', '_r')
        new_s = new_s.replace('_w_', '_w')
        new_s = new_s.replace('_b_', '_b')
        mapping[s] = new_s
    mapping["clock_uncore"] = "clock"
    mapping["reset_io"] = "reset"
    sorted_mapping = sorted(mapping.items(), key=lambda x: -len(x[0]))

    # 替换整个文件内容
    full_content = ''.join(content)
    for old, new in sorted_mapping:
        full_content = full_content.replace(old, new)

    # 写入新文件
    with open(output_path, 'w') as f:
        f.write(full_content)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_file")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_file(input_file, output_file)
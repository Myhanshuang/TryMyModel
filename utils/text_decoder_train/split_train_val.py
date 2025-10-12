# split_data.py
import random

input_file = "/home/acacia/PycharmProjects/PythonProject5/data/flickr8k_captions.txt"
train_file = "/home/acacia/PycharmProjects/PythonProject5/data/train_captions.txt"
val_file = "/home/acacia/PycharmProjects/PythonProject5/data/val_captions.txt"
split_ratio = 0.9  # 90% 的数据用于训练，10% 用于验证

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 打乱数据顺序
    random.shuffle(lines)

    # 计算分割点
    split_point = int(len(lines) * split_ratio)

    # 分割数据
    train_lines = lines[:split_point]
    val_lines = lines[split_point:]

    # 写入训练集文件
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # 写入验证集文件
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print(f"数据分割完成！")
    print(f"训练数据: {len(train_lines)} 条，已保存至 {train_file}")
    print(f"验证数据: {len(val_lines)} 条，已保存至 {val_file}")

except FileNotFoundError:
    print(f"错误: 未找到输入文件 '{input_file}'。请先运行数据提取脚本。")
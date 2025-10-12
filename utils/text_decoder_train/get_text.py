import json
import os


def prepare_pretraining_data_correctly(input_json_path, train_output_path, val_output_path):
    """
    读取原始的 Flickr8k JSON 数据，并根据 "split" 字段严格分离训练集和验证集
    的文本，完全排除测试集。

    Args:
        input_json_path (str): 原始 dataset_flickr8k.json 文件的路径。
        train_output_path (str): 用于预训练的训练集文本输出路径。
        val_output_path (str): 用于预训练的验证集文本输出路径。
    """
    print(f"正在从 {input_json_path} 读取数据并进行严格分割...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_json_path}")
        return

    train_sentences = []
    val_sentences = []
    test_sentences_count = 0

    # 遍历原始数据中的每一张图片
    for image_data in original_data.get("images", []):
        split = image_data.get("split")
        sentences = [s.get("raw") for s in image_data.get("sentences", []) if s.get("raw")]

        if split == "train":
            train_sentences.extend(sentences)
        elif split == "val":  # 部分数据集可能称之为 "dev" 或 "validation"
            val_sentences.extend(sentences)
        elif split == "test":
            test_sentences_count += len(sentences)
        # 其他 split (如果有的话) 将被忽略

    print("\n数据分割统计:")
    print(f" - 训练集句子数量: {len(train_sentences)}")
    print(f" - 验证集句子数量: {len(val_sentences)}")
    print(f" - 测试集句子数量 (已忽略): {test_sentences_count}")

    # 写入训练集文件
    try:
        with open(train_output_path, 'w', encoding='utf-8') as f:
            for sentence in train_sentences:
                f.write(sentence + '\n')
        print(f"\n成功将训练集文本写入到: {train_output_path}")
    except IOError as e:
        print(f"写入训练文件时出错: {e}")

    # 写入验证集文件
    try:
        with open(val_output_path, 'w', encoding='utf-8') as f:
            for sentence in val_sentences:
                f.write(sentence + '\n')
        print(f"成功将验证集文本写入到: {val_output_path}")
    except IOError as e:
        print(f"写入验证文件时出错: {e}")


if __name__ == '__main__':
    # 定义输入和输出文件路径
    input_file = "/home/acacia/PycharmProjects/PythonProject5/data/flickr8k_aim3/dataset_flickr8k.json"
    train_output_file = "/home/acacia/PycharmProjects/PythonProject5/data/train_captions.txt"
    val_output_file = "/home/acacia/PycharmProjects/PythonProject5/data/val_captions.txt"

    # 运行转换函数
    prepare_pretraining_data_correctly(input_file, train_output_file, val_output_file)
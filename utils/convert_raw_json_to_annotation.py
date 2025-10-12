import json
import os
from datetime import datetime


def convert_flickr8k_to_coco_format(input_json_path, output_json_path):
    """
    读取原始的 Flickr8k JSON 数据，筛选出测试集，
    并将其转换为包含 'info' 和 'licenses' 的完整 COCO 标注格式。

    Args:
        input_json_path (str): 原始 dataset_flickr8k.json 文件的路径。
        output_json_path (str): 转换后要保存的 COCO 格式 JSON 文件的路径。
    """
    print(f"正在从 {input_json_path} 读取数据...")
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_json_path}")
        return

    # --- 关键修改：添加 info 和 licenses 字段 ---
    # COCO 格式的完整基本结构
    coco_format_data = {
        "info": {
            "description": "Flickr8k Test Set in COCO format",
            "url": "N/A",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Conversion Script",
            "date_created": datetime.now().strftime('%Y/%m/%d')
        },
        "licenses": [],  # 通常可以留空
        "images": [],
        "annotations": [],
        "type": "captions"
    }
    # --- 修改结束 ---

    print("开始转换数据格式...")
    annotation_id_counter = 1
    test_image_count = 0

    for image_data in original_data.get("images", []):
        if image_data.get("split") == "test":
            test_image_count += 1

            image_info = {
                "id": image_data.get("imgid"),
                "file_name": image_data.get("filename"),
                "license": 0,  # 添加一个占位符
                "height": 0,  # 通常这些元数据是可选的，但最好有占位符
                "width": 0,
            }
            coco_format_data["images"].append(image_info)

            for sentence in image_data.get("sentences", []):
                annotation_info = {
                    "image_id": image_data.get("imgid"),
                    "id": annotation_id_counter,
                    "caption": sentence.get("raw")
                }
                coco_format_data["annotations"].append(annotation_info)
                annotation_id_counter += 1

    print(f"处理完成。共找到 {test_image_count} 张测试集图片。")
    print(f"共生成 {annotation_id_counter - 1} 条标注信息。")

    print(f"正在将结果保存到 {output_json_path} ...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format_data, f, indent=4)

    print("文件保存成功！")


if __name__ == '__main__':
    # 定义输入和输出文件路径
    # 确保 dataset_flickr8k.json 文件与此脚本位于同一目录
    input_file = "/home/acacia/PycharmProjects/PythonProject5/data/flickr8k_aim3/dataset_flickr8k.json"
    output_file = "/home/acacia/PycharmProjects/PythonProject5/flickr8k_test_coco_format.json"

    # 运行转换函数
    convert_flickr8k_to_coco_format(input_file, output_file)
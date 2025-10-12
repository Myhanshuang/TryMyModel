import json
from collections import Counter

# --- 请将 'your_results_file.json' 替换为您的预测结果文件名 ---
RESULTS_FILE = '/home/acacia/PycharmProjects/PythonProject5/my_predictions.json'


def find_duplicate_image_ids(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {json_file}")
        return

    image_ids = [item['image_id'] for item in data]

    id_counts = Counter(image_ids)

    duplicates = {img_id: count for img_id, count in id_counts.items() if count > 1}

    if not duplicates:
        print("检查完成：文件中没有重复的 image_id。")
    else:
        print("错误：在文件中发现以下重复的 image_id：")
        for img_id, count in duplicates.items():
            print(f"  - Image ID: {img_id}, 出现了 {count} 次")


if __name__ == '__main__':
    find_duplicate_image_ids(RESULTS_FILE)
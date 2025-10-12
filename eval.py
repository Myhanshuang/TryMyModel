import sys
import json

# 1. 设置 coco-caption 库的路径
#    请将 'path/to/your/coco-caption' 替换为您的实际路径
COCO_CAPTION_PATH = '/home/acacia/PycharmProjects/PythonProject5/utils/coco-caption'

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def evaluate(annotation_file, results_file, model_name):
    # 3. 创建 COCO 对象并加载结果
    #    COCO(...) 会加载并解析标注文件
    sys.path.append(COCO_CAPTION_PATH)
    coco = COCO(annotation_file)
    #    coco.loadRes(...) 会加载模型生成的结果文件
    coco_results = coco.loadRes(results_file)

    # 4. 创建评估对象
    coco_eval = COCOEvalCap(coco, coco_results)

    # 5. 设置要评估的图片ID
    #    可以评估结果文件中包含的所有图片
    coco_eval.params['image_id'] = coco_results.getImgIds()

    # 6. 执行评估
    coco_eval.evaluate()

    # 7. 打印所有评估指标的结果
    print("\nEvaluation Metrics:")
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.4f}')

    # 您也可以将结果保存到 JSON 文件
    results_summary = {metric: score for metric, score in coco_eval.eval.items()}
    with open(f'{model_name}_evaluation_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)

    print("\nEvaluation summary saved to evaluation_summary.json")
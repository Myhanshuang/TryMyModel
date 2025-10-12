
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile  # 用于创建临时文件

import torch
import json


def generate_predictions(model, dataloader, tokenizer, device, output_filename='my_prediction'):

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['pixel_values']
            image_ids = batch['img_id']
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images = images.to(device)

            # generated_ids = model.beam_generate(
            #     images,
            #     sos_token_id=tokenizer.cls_token_id,
            #     eos_token_id=tokenizer.sep_token_id,
            #     max_length=50,
            #     beam_size=5
            # )

            generated_ids = model.generate(
                pixel_values=images,
                num_beams=5,
                max_length=50,
                pad_token_id=tokenizer.pad_token_id
            )

            generated_captions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            cleaned_captions = [cap.strip() for cap in generated_captions]
            if isinstance(image_ids, int):
                image_ids = [image_ids]
            for img_id, caption in zip(image_ids, cleaned_captions):
                if img_id not in [pred['image_id'] for pred in predictions]:
                    predictions.append({
                        'image_id': img_id,
                        'caption': caption
                    })
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        print(f"预测结果已成功保存到: {output_filename}")
    except Exception as e:
        print(f"写入 JSON 文件时出错: {e}")

    return predictions

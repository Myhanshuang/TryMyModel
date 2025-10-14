import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .dataset import build_paper_dataset
from .model import PaperModel, SimpleTokenizer
from .train import train
from eval import evaluate
import sys
import os
sys.path.append('/home/acacia/PycharmProjects/PythonProject5')
IMG_DIR = "/home/acacia/PycharmProjects/PythonProject5/data/flickr8k_aim3/images"
JSON_PATH = "/home/acacia/PycharmProjects/PythonProject5/data/flickr8k_aim3/dataset_flickr8k.json"
SAVE_DIR = "/home/acacia/PycharmProjects/PythonProject5/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-6
MAX_LENGTH = 50
D_MODEL = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(JSON_PATH, 'r') as f:
    data = json.load(f)
captions = [s['raw'] for img in data['images'] for s in img['sentences']]
tokenizer = SimpleTokenizer(captions=captions, freq_threshold=5, max_len=50, path='/home/acacia/PycharmProjects/PythonProject5/paper/tokenizer.json')

model = PaperModel(hidden_state=D_MODEL, encoder_dim=2048, device=DEVICE, vocab_size=tokenizer.vocab_size, dropout=0.5)

model.to(DEVICE)

pad_id = tokenizer.pad_token_id

train_dataset = build_paper_dataset(IMG_DIR, tokenizer, JSON_PATH, split="train")
test_dataset = build_paper_dataset(IMG_DIR, tokenizer, JSON_PATH, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# optimizer = optim.AdamW([{'params': model.encoder.parameters(),'lr' : LR},
#                         {'params': model.decoder.parameters(), 'lr' : LR * 0.1}])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS)

best_loss = train(
    epochs=EPOCHS,
    model=model,
    loss=loss_fn,
    optimizer=optimizer,
    trainloader=train_loader,
    testloader=test_loader,
    path_dir=SAVE_DIR,
    pad_id=pad_id,
    device=DEVICE,
    scheduler=scheduler
)

print('-------------------------------------')
model.eval()
results = []

test_dataset = build_paper_dataset(IMG_DIR, tokenizer, JSON_PATH, split="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
with torch.no_grad():
    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        img_ids = batch["img_id"]
        output_ids = model.generate(
            pixel_values=pixel_values,
            max_length=MAX_LENGTH,
            num_beams=4
        )
        captions = [tokenizer.decode(ids.cpu().tolist()) for ids in output_ids]

        for img_id, caption in zip(img_ids, captions):
            results.append({
                "image_id": img_id.item(),
                "caption": caption
            })


results_file = os.path.join(SAVE_DIR, "results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Saved results to {results_file}")

ANNOTATION_FILE = '/home/acacia/PycharmProjects/PythonProject5/flickr8k_test_coco_format.json'
print("Running COCO Caption evaluation...")
evaluate(annotation_file=ANNOTATION_FILE, results_file=results_file, model_name=model.name)

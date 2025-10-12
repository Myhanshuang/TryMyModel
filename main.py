import os
from peft import LoraConfig, get_peft_model, TaskType # 导入 PEFT 相关库
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, VisionEncoderDecoderModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from utils.checkpoint import load_best_model
from dataset import build_dataset, build_k_fold_dataset, build_dataset_pretrain
from model.img_encoder.ViT_encoder import ViTEncoder
from model.img_encoder.my_encoder import MyVisionEncoder
from model.text_decoder.my_decoder import MyTextDecoder
from model.model import Model, ShowAttendTellModel
from train import train
from predict import generate_predictions
from eval import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_DIR = "data/flickr8k_aim3/images"
JSON_PATH = "data/flickr8k_aim3/dataset_flickr8k.json"
CHECKPOINT_DIR = "/home/acacia/PycharmProjects/PythonProject5/checkpoints"

D_MODEL = 128
NHEAD = 8
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
seed = 42
weight_decay = 0
annotation = 'flickr8k_test_coco_format.json'
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE = tokenizer.vocab_size
output_filename = 'my_predictions.json'


def get_model(D_MODEL, VOCAB_SIZE, NHEAD, DEVICE):
    encoder = MyVisionEncoder(output_dim=D_MODEL)
    # encoder = ViTEncoder(d_model=D_MODEL, output_dim=D_MODEL, n_layers=4)
    decoder = MyTextDecoder(d_model=D_MODEL, vocab_size=VOCAB_SIZE, nhead=NHEAD)
    model = Model(img_encoder=encoder, text_decoder=decoder, name='MyModel').to(DEVICE)

    # model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    #     "google/vit-base-patch16-224-in21k",
    #     "gpt2"
    # ).to(DEVICE)
    # model.config.decoder_start_token_id = tokenizer.bos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["query", "value", "q_proj", "v_proj", "c_attn"],
    #     lora_dropout=0.05,
    #     bias="none",  # 通常不训练 bias
    # )
    # peft_model = get_peft_model(model, lora_config)

    # model = ShowAttendTellModel(d_model=D_MODEL, embed_dim=D_MODEL, decoder_dim=D_MODEL, vocab_size=tokenizer.vocab_size).to(DEVICE)
    return model


def main():
    scores = normal_train(
        IMG_DIR=IMG_DIR,
        JSON_PATH=JSON_PATH,
        BATCH_SIZE=BATCH_SIZE,
        D_MODEL=D_MODEL,
        NHEAD=NHEAD,
        DEVICE=DEVICE,
        LEARNING_RATE=LEARNING_RATE,
        EPOCHS=EPOCHS,
        CHECKPOINT_DIR=CHECKPOINT_DIR,
        weight_decay=weight_decay,
        tokenizer=tokenizer,
        num_warmup_steps=6,
        num_training_steps=20
    )

    # scores = run_k_fold_training(
    #     img_dir=IMG_DIR,
    #     json_path=JSON_PATH,
    #     device=DEVICE,
    #     checkpoint_dir=CHECKPOINT_DIR,
    #     n_splits=5,
    #     batch_size=BATCH_SIZE,
    #     d_model=D_MODEL,
    #     nhead=NHEAD,
    #     lr=LEARNING_RATE,
    #     epochs=EPOCHS,
    #     seed=seed,
    #     weight_decay=weight_decay,
    #     tokenizer=tokenizer
    # )
    # print(scores)
    model_name = 'MyModel'
    model = get_model(D_MODEL=D_MODEL, VOCAB_SIZE=VOCAB_SIZE, NHEAD=NHEAD, DEVICE=DEVICE)
    model = load_best_model(CHECKPOINT_DIR, model)
    test_loader = build_dataset(img_dir=IMG_DIR, tokenizer=tokenizer, json_path=JSON_PATH, split='test')
    # test_loader = build_dataset(img_dir=IMG_DIR, tokenizer=tokenizer, json_path=JSON_PATH, split='test')
    # outputs = generate_predictions(model, test_loader, tokenizer, DEVICE, model.name + output_filename)
    # evaluate(annotation, model.name + output_filename, model.name)
    outputs = generate_predictions(model, test_loader, tokenizer, DEVICE, model_name + output_filename)
    evaluate(annotation, model_name + output_filename, model_name)

def normal_train(tokenizer, IMG_DIR, JSON_PATH, BATCH_SIZE, D_MODEL, NHEAD, DEVICE, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR, weight_decay, num_warmup_steps, num_training_steps):
    VOCAB_SIZE = tokenizer.vocab_size
    train_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='train')
    val_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='val')
    # train_dataset = build_dataset_pretrain(IMG_DIR, tokenizer, JSON_PATH, split='train')
    # val_dataset = build_dataset_pretrain(IMG_DIR, tokenizer, JSON_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True)

    model = get_model(D_MODEL=D_MODEL, NHEAD=NHEAD, DEVICE=DEVICE, VOCAB_SIZE=VOCAB_SIZE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    return train(
        epochs=EPOCHS,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        trainloader=train_loader,
        testloader=val_loader,
        path_dir=CHECKPOINT_DIR,
        pad_id=tokenizer.pad_token_id,
        device=DEVICE,
        # scheduler=scheduler
    )

def run_k_fold_training(tokenizer, img_dir, json_path, device, checkpoint_dir, n_splits, batch_size, d_model, nhead, lr, epochs, seed, weight_decay):
    vocab_size = tokenizer.vocab_size
    all_fold_scores = []
    all_datasets = build_k_fold_dataset(img_dir=img_dir, tokenizer=tokenizer, json_path=json_path, n_splits=n_splits, seed=seed)
    for fold_idx, (train_dataset, val_dataset) in enumerate(all_datasets):
        print(f"训练第 {fold_idx + 1}/{n_splits} 折")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        model = get_model(D_MODEL=d_model, VOCAB_SIZE=vocab_size, NHEAD=nhead, DEVICE=device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        fold_checkpoint_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)

        best_loss = train(
            epochs=epochs,
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            trainloader=train_loader,
            testloader=val_loader,
            path_dir=fold_checkpoint_dir,
            device=device,
            pad_id=tokenizer.pad_token_id
        )

        print(f"第 {fold_idx + 1} 折训练完成。最佳验证损失: {best_loss:.4f}")
        all_fold_scores.append(best_loss)

    return all_fold_scores

def train_decoder(tokenizer, IMG_DIR, JSON_PATH, BATCH_SIZE, D_MODEL, NHEAD, DEVICE, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR, weight_decay):
    VOCAB_SIZE = tokenizer.vocab_size
    train_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='train')
    val_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True)

    model = get_model(D_MODEL=D_MODEL, NHEAD=NHEAD, DEVICE=DEVICE, VOCAB_SIZE=VOCAB_SIZE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(
        [{'params': model.decoder.parameters(), 'lr': LEARNING_RATE}],
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )
    return train(
        epochs=EPOCHS,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        trainloader=train_loader,
        testloader=val_loader,
        path_dir=CHECKPOINT_DIR,
        pad_id=tokenizer.pad_token_id,
        device=DEVICE
    )

def train_with_pretrained(tokenizer, IMG_DIR, JSON_PATH, BATCH_SIZE, D_MODEL, NHEAD, DEVICE, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR, weight_decay):
    VOCAB_SIZE = tokenizer.vocab_size
    train_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='train')
    val_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, persistent_workers=True)

    model = get_model(D_MODEL=D_MODEL, NHEAD=NHEAD, DEVICE=DEVICE, VOCAB_SIZE=VOCAB_SIZE)
    # all_para = torch.load('/home/acacia/PycharmProjects/PythonProject5/checkpoints/try12_consistence/stage2/ViTModel_latest_model.pth', weights_only=False)
    # model.load_state_dict(all_para['model_state_dict'])
    # pre_trained_decoder = torch.load('/home/acacia/PycharmProjects/PythonProject5/utils/text_decoder_train/best_pretrained_decoder.pth')
    # model.decoder.load_state_dict(pre_trained_decoder)
    # print(model.decoder.alpha.item())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    vision_param, text_param, mixed_param = para_list(model)

    optimizer = torch.optim.Adam(
        [{'params': vision_param, 'lr': LEARNING_RATE * 10},
         {'params': text_param, 'lr': LEARNING_RATE * 0.1},
         {'params': mixed_param, 'lr': LEARNING_RATE}],
        weight_decay=weight_decay
    )
    return train(
        epochs=EPOCHS,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        trainloader=train_loader,
        testloader=val_loader,
        path_dir=CHECKPOINT_DIR,
        pad_id=tokenizer.pad_token_id,
        device=DEVICE
    )

def para_list(model):

    vision_related_params = []
    text_logic_params = []
    mixed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'encoder' in name or 'decoder.multihead_attn' in name or 'alpha' in name:
            vision_related_params.append(param)

        elif 'decoder.emb' in name or 'decoder.self_attn' in name or 'decoder.out_fc' in name:
            text_logic_params.append(param)

        else:
            mixed_params.append(param)

    total_params = len(list(model.parameters()))
    grouped_params = len(vision_related_params) + len(text_logic_params) + len(mixed_params)
    assert total_params == grouped_params, "Some parameters were not assigned to any group!"

    print(f"\nGrouping complete. Total params: {total_params}, Grouped params: {grouped_params}")
    return vision_related_params, text_logic_params, mixed_params

def split_train(tokenizer, IMG_DIR, JSON_PATH, BATCH_SIZE, D_MODEL, NHEAD, DEVICE, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR, weight_decay):
    VOCAB_SIZE = tokenizer.vocab_size
    train_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='train')
    val_dataset = build_dataset(IMG_DIR, tokenizer, JSON_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True)

    model = get_model(D_MODEL=D_MODEL, NHEAD=NHEAD, DEVICE=DEVICE, VOCAB_SIZE=VOCAB_SIZE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(
        [{'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 10},
         {'params': model.decoder.parameters(), 'lr':LEARNING_RATE}],
        weight_decay=weight_decay)

    return train(
        epochs=EPOCHS,
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        trainloader=train_loader,
        testloader=val_loader,
        path_dir=CHECKPOINT_DIR,
        pad_id=tokenizer.pad_token_id,
        device=DEVICE
    )

if __name__ == '__main__':
    main()

import torch.nn
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from model.text_decoder.my_decoder import MyTextDecoder
from main import D_MODEL, tokenizer, NHEAD, CHECKPOINT_DIR, BATCH_SIZE, DEVICE
text_decoder = MyTextDecoder(d_model = D_MODEL, vocab_size=tokenizer.vocab_size, nhead=NHEAD)

train_dataset = TextDataset("/home/acacia/PycharmProjects/PythonProject5/data/train_captions.txt", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # 训练集需要打乱

val_dataset = TextDataset("/home/acacia/PycharmProjects/PythonProject5/data/val_captions.txt", tokenizer)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # 验证集不需要打乱


optimizer = torch.optim.Adam(text_decoder.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
num_epochs = 10
text_decoder.to(DEVICE)
text_decoder.train()


def evaluate(model, dataloader, loss_fn, device):
    """
    在验证集上评估模型性能。
    """
    # 将模型设置为评估模式
    model.eval()
    total_loss = 0

    # 在评估时，我们不需要计算梯度
    with torch.no_grad():
        for input_ids in dataloader:
            input_ids = input_ids.to(device)

            # 准备输入和标签
            model_inputs = input_ids[:, :-1].contiguous()
            labels = input_ids[:, 1:].contiguous()
            batch_size = model_inputs.shape[0]
            d_model = model.emb.embedding_dim
            num_patches = 50
            dummy_img = torch.zeros((batch_size, num_patches, d_model), device=device)
            # 前向传播
            outputs = model(dummy_img, model_inputs)

            # 计算损失
            loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))
            total_loss += loss.item()

    # 返回整个验证集的平均损失
    return total_loss / len(dataloader)


best_val_loss = float('inf')  # 用于记录最佳验证损失
model_save_path = "best_pretrained_decoder.pth"

for epoch in range(num_epochs):
    # --- 训练阶段 ---
    text_decoder.train()  # 将模型设置为训练模式
    total_train_loss = 0

    # 使用 tqdm 显示进度条
    # from tqdm import tqdm
    # train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for input_ids in train_loader:  # train_iterator
        input_ids = input_ids.to(DEVICE)

        model_inputs = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()

        batch_size = model_inputs.shape[0]

        # 获取模型的维度 d_model
        d_model = text_decoder.emb.embedding_dim

        # 创建一个全零的伪图像张量。
        # 形状需要匹配交叉注意力期望的 memory 形状: (batch_size, num_patches, d_model)
        # num_patches 可以是任意值，例如 50
        num_patches = 50
        dummy_img = torch.zeros((batch_size, num_patches, d_model), device=DEVICE)

        optimizer.zero_grad()
        outputs = text_decoder(dummy_img, model_inputs)
        loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- 评估阶段 ---
    # 调用我们上面定义的 evaluate 函数
    avg_val_loss = evaluate(text_decoder, val_loader, loss_fn, DEVICE)

    # 打印每个 epoch 的结果
    print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- 保存最佳模型 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(text_decoder.state_dict(), model_save_path)
        print(f"Validation loss improved. Model saved to {model_save_path}")

print("\n预训练完成！最佳模型已保存。")
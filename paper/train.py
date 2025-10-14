from torch.cuda.amp import autocast, GradScaler
from utils.checkpoint import save_checkpoint, load_checkpoint
import json
import torch
import matplotlib.pyplot as plt
import os
import time
def set_pic_config():
    # --- 动态绘图设置 ---
    # 1. 开启交互模式
    plt.ion()
    # 2. 创建一个图形和一个子图 (axes)
    fig, ax = plt.subplots()
    # 3. 初始化两条空的曲线，并获取它们的引用
    train_line, = ax.plot([], [], 'r-o', label='Train Loss')
    test_line, = ax.plot([], [], 'b-x', label='Test Loss')
    # 4. 设置图表的属性
    ax.set_title('Training and Test Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    # --- 结束设置 ---
    return train_line, test_line, ax, fig

def update_pic(train_loss_all, test_loss_all, train_line, test_line, ax, fig):
    # --- 动态更新图表 ---
    epochs_ran = range(len(train_loss_all))
    # 5. 更新曲线的数据
    train_line.set_xdata(epochs_ran)
    train_line.set_ydata(train_loss_all)
    test_line.set_xdata(epochs_ran)
    test_line.set_ydata(test_loss_all)

    # 6. 重新计算坐标轴范围并刷新画布
    ax.relim()
    ax.autoscale_view()
    fig.canvas.flush_events()
    # --- 更新结束 ---

def finish_pic(path_dir, fig, model_name):
    # --- 训练结束后关闭交互模式 ---
    plt.ioff()
    # 保存最终的图像
    final_plot_path = f"{path_dir}/{model_name}loss_curve.png"
    fig.savefig(final_plot_path)
    print(f"Final loss curve saved to {final_plot_path}")
    # 显示最终图像，程序会在此暂停直到关闭窗口
    plt.show()

def evaluate(model, loss, dataloader, pad_id, device):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img = batch['pixel_values'].to(device)
            tgt = batch['labels'].to(device)
            pred, _ = model(img, tgt)
            label = tgt[:, 1:]
            loss_num = loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            total_loss += loss_num.item()

    return total_loss / len(dataloader)

def train_one_epoch(model, dataloader, loss, epoch, optimizer, pad_id, device, scaler, scheduler = None, lmd=0.01):
    total_loss = 0
    model.train()
    print(f"number {epoch}th training :")
    for i, batch in enumerate(dataloader):
        img = batch['pixel_values'].to(device)
        tgt = batch['labels'].to(device)
        label = tgt[:, 1:]
        optimizer.zero_grad()
        with autocast():
            pred, alpha = model(img, tgt)
            print(alpha.std().item())
            att_reg = lmd * ((1 - alpha.sum(dim=1)) ** 2).mean()
            loss_num = loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1)) + att_reg
        total_loss += loss_num.item()
        scaler.scale(loss_num).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
    return total_loss / len(dataloader)

def train(epochs, model, loss, optimizer, trainloader, testloader, path_dir, pad_id, device, scheduler=None):
    train_loss_all, test_loss_all = [], []

    train_line, test_line, ax, fig = set_pic_config()
    scaler = GradScaler()
    start_epoch = 0
    best_models = []
    if path_dir is not None:
        start_epoch, best_models, train_loss_all, test_loss_all = load_checkpoint(path_dir=path_dir, model=model, optimizer=optimizer, scheduler=scheduler)

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model=model, dataloader=trainloader, loss=loss, epoch=epoch, optimizer=optimizer, scheduler=scheduler, pad_id=pad_id, device=device, scaler=scaler)
        test_loss = evaluate(model=model, dataloader=testloader, loss=loss, pad_id=pad_id, device=device)
        end_time = time.time()

        train_loss_all.append(train_loss)
        test_loss_all.append(test_loss)
        print(f'Epoch{epoch}/{epochs}: trainloss:{train_loss}, testloss:{test_loss}, using {end_time - start_time}')

        update_pic(train_loss_all, test_loss_all, train_line, test_line, ax, fig)

        best_models = save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, loss=test_loss, path_dir=path_dir, best_models=best_models, scheduler=scheduler, train_loss_all=train_loss_all, test_loss_all=test_loss_all)

    finish_pic(path_dir=path_dir, fig=fig, model_name=model.name)

    return min(test_loss_all)

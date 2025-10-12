import torch
import os
import json

def save_checkpoint(model, optimizer, epoch, loss, best_models, path_dir, train_loss_all, test_loss_all, max_models=3, scheduler=None):
    # if path_dir is None:
    #     return

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, exist_ok=True)
    model_name = model.name
    # model_name = 'pretrain_vit_gpt2'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'loss': loss,
        'model_name': model_name
    }
    model_path = os.path.join(path_dir, f"{model_name}_latest_model.pth")
    torch.save(checkpoint, model_path)
    print(f"Latest Model saved at {model_path}")

    model_path = os.path.join(path_dir, f"{model_name}_epoch{epoch}_loss_{loss:.4f}.pth")
    torch.save(checkpoint, model_path)
    best_models.append((loss, model_path))
    best_models.sort(key=lambda x: x[0])

    if len(best_models) > max_models:
        _, worst_model_path = best_models.pop()
        if os.path.exists(worst_model_path):
            os.remove(worst_model_path)
            print(f"Removed worst model: {worst_model_path}")

    history_data = {'train_loss': train_loss_all, 'test_loss': test_loss_all}
    history_path = os.path.join(path_dir, f'{model_name}_loss_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_data, f)

    return best_models

def load_checkpoint(path_dir, model, optimizer, scheduler=None):
    # model_name = model.name

    model_name = 'pretrain_vit_gpt2'
    checkpoint_file = None
    if not os.path.exists(path_dir):
        return 0, [], [], []
    for file in os.listdir(path_dir):
        if file.startswith(f"{model_name}_latest_model.pth"):
            checkpoint_file = os.path.join(path_dir, file)
            break

    if not checkpoint_file:
        print(f"No checkpoint found for model {model_name} in {path_dir}")
        return 0, [], [], []

    checkpoint = torch.load(checkpoint_file, weights_only=False)
    # if model_name != checkpoint['model_name']:
    #     raise RuntimeError('ERROR!!!: model does not match the weights')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {path_dir}, resuming from epoch {start_epoch}")

    history_path = os.path.join(path_dir, f'{model_name}_loss_history.json')
    train_loss_all, test_loss_all = [], []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_data = json.load(f)
            train_loss_all = history_data.get('train_loss', [])
            test_loss_all = history_data.get('test_loss', [])
        print("Loaded loss history.")

    best_models = []
    for file in os.listdir(path_dir):
        if file.startswith(f"{model_name}_epoch") and file.endswith(".pth"):
            try:
                loss_str = file.split('_loss_')[-1].replace('.pth', '')
                loss = float(loss_str)
                best_models.append((loss, os.path.join(path_dir, file)))
            except (ValueError, IndexError):
                continue

    best_models.sort(key=lambda x: x[0])
    print(f"Reconstructed best_models list with {len(best_models)} models.")


    return start_epoch, best_models, train_loss_all, test_loss_all


def load_best_model(path_dir, model):

    _, best_models, _, _ = load_checkpoint(path_dir, model, optimizer=None)

    if not best_models:
        raise FileNotFoundError("No models found in the checkpoint directory.")

    best_model_path = best_models[0][1]
    print(f"Loading best model from: {best_model_path}")

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
import torch


def save_model(save_path, iteration, optimizer, model):
    torch.save({'iteration': iteration,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()}, save_path)


def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])

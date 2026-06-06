import torch

def ewc_loss(model, fisher_matrix, lambda_ewc):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_matrix:
            loss += (fisher_matrix[name] * (param - model.initial_params[name])**2).sum()
    return lambda_ewc * loss

def compute_fisher(model, dataloader):
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    
    model.eval()
    for x, y in dataloader:
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        grad = torch.autograd.grad(torch.log(prob[:, y]).sum(dim=0))
        fisher[name] += grad**2  # 简化计算
    
    return {name: f/len(dataloader) for name, f in fisher.items()}
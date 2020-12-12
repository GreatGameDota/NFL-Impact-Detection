import torch

def get_optimizer(model, lr=0.001, lookahead=False):
    # base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0004)
    
    return optimizer
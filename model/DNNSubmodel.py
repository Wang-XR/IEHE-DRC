import torch
import numpy as np

class BPNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(BPNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100, 50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_nn_model(cluster_data, args, model_name, loss_name, optimizer_name, batch_size, epochs):
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    g = torch.Generator()
    g.manual_seed(args.random_state)
    model_factory = {
        "bpnn": lambda: BPNN(len(args.features)),
    }
    if model_name not in model_factory:
        raise ValueError(f"Unknown model_name: {model_name}")
    model = model_factory[model_name]()

    loss_factory = {
        "mse": torch.nn.MSELoss,
    }
    if loss_name not in loss_factory:
        raise ValueError(f"Unknown loss_name: {loss_name}")
    criterion = loss_factory[loss_name]()

    optimizer_factory = {
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4),
    }
    if optimizer_name not in optimizer_factory:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")
    optimizer = optimizer_factory[optimizer_name]()

    X_train = cluster_data[args.features].values
    y_train = cluster_data[args.target].values
    X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype = torch.float32).view(-1, 1)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, generator = g)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model
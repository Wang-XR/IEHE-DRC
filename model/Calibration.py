import torch
import numpy as np

class Gate(torch.nn.Module):
    def __init__(self, features_dim, unsupervisedensemble, lambda_fusion = 1, lambda_rank = 0.1,
                 delta = 0.1, xi = 1.0, Temperature = 5, random_state = 42, use_xavier_init = False):
        super(Gate, self).__init__()

        self.features_dim = features_dim
        self.num_experts = len(unsupervisedensemble.experts)
        self.unsupervisedensemble = unsupervisedensemble
        self.lambda_fusion = lambda_fusion
        self.lambda_rank = lambda_rank
        self.delta = delta
        self.xi = xi
        self.random_state = random_state
        self.Temperature = Temperature
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(features_dim, 50),
            torch.nn.Sigmoid(),
            torch.nn.Linear(50, 25),
            torch.nn.Sigmoid(),
            torch.nn.Linear(25, self.num_experts),
            torch.nn.Softmax(dim=1)
        )

        self.W_u = torch.nn.Linear(2 * self.num_experts, self.num_experts)
        self.W_r = torch.nn.Linear(2 * self.num_experts, self.num_experts)
        if use_xavier_init:
            self._init_gate_weights_xavier()

    def _init_gate_weights_xavier(self):
        torch.nn.init.xavier_uniform_(self.W_u.weight)
        torch.nn.init.xavier_uniform_(self.W_r.weight)

        if self.W_u.bias is not None:
            torch.nn.init.zeros_(self.W_u.bias)
        if self.W_r.bias is not None:
            torch.nn.init.zeros_(self.W_r.bias)


    def rank_loss(self, adjusted_weights, submodel_predictions, Y):
        n_samples, n_experts = submodel_predictions.shape
        errors = torch.abs(submodel_predictions - Y.view(-1, 1))
        best_model_idx = torch.argmin(errors, dim=1)
        w_best = adjusted_weights[torch.arange(n_samples), best_model_idx]
        loss_best = ((1 - w_best) ** 2).mean()
        mask = torch.ones_like(adjusted_weights, dtype=torch.bool)
        mask[torch.arange(n_samples), best_model_idx] = False
        loss_margin = torch.nn.functional.relu(adjusted_weights[mask].view(n_samples, n_experts - 1) - w_best.unsqueeze(1) + self.delta).mean()
        loss_total = loss_best + self.xi * loss_margin
        return loss_total

    def fusion_loss(self, predictions, Y):
        return torch.nn.functional.mse_loss(predictions, Y)

    def fit(self, X, Y, optimizer, epochs = 100, batch_size = 256):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        g = torch.Generator()
        g.manual_seed(self.random_state)
        optimizer = optimizer
        dataset = torch.utils.data.TensorDataset(X, Y)
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size = batch_size,
                                                      shuffle = True,
                                                      drop_last = False,
                                                      generator = g)
        for epoch in range(epochs):
            self.train()
            running_loss = 0
            for i, (X_batch, Y_batch) in enumerate(dataset_loader):
                optimizer.zero_grad()
                loss = self.loss(X_batch, Y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss
            average_loss = round((running_loss.detach().numpy() / (i + 1)), 5)
            print(f"Epoch {epoch + 1} average loss: {average_loss}")
        return

    def loss(self, X, Y):
        submodel_predictions, predictions, adjusted_weights = self.forward(X)
        loss = self.lambda_fusion * self.fusion_loss(predictions, Y) + self.lambda_rank * self.rank_loss(adjusted_weights, submodel_predictions, Y)
        return loss

    def forward(self, X):
        self.unsupervisedensemble.eval()
        with torch.no_grad():
            submodel_predictions, pretraining_predictions, weights = self.unsupervisedensemble.predict(X)
        delta_weights = self.mlp(X)
        combination = torch.cat((weights, delta_weights), dim=1)

        update_gate = torch.nn.functional.relu(self.W_u(combination))
        reset_gate = torch.nn.functional.relu(self.W_r(combination))

        adjusted_weights = update_gate * weights + (1 - update_gate) * (reset_gate * delta_weights)
        if self.training:
            adjusted_weights = torch.softmax(adjusted_weights / self.Temperature, dim=1)
        else:
            adjusted_weights = torch.softmax(adjusted_weights, dim=1)
        assert submodel_predictions.shape == adjusted_weights.shape, (
            f"Shape mismatch: submodel_predictions {submodel_predictions.shape}, "
            f"adjusted weights {adjusted_weights.shape}"
        )

        predictions = (submodel_predictions * adjusted_weights).sum(axis=1)

        return submodel_predictions, predictions, adjusted_weights
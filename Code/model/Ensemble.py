import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import joblib
import os

class DataDensity:
    def __init__(self, mean, cov, random_state):
        """
        Parameters
        ----------
        mean : array-like, shape (d,)
        cov  : array-like
            - shape (d,)   : treated as diagonal variances
            - shape (d, d) : treated as full covariance
        """

        self.random_state = random_state
        self.mean = np.asarray(mean)
        cov = np.asarray(cov)

        if cov.ndim == 1:
            self.cov = np.diag(cov)
        elif cov.ndim == 2:
            self.cov = cov
        else:
            raise ValueError(
                f"`var` must be 1D or 2D array, got shape {cov.shape}"
            )

        self.density = multivariate_normal(
            mean = self.mean,
            cov = self.cov,
            allow_singular = True,
        )

    def pdf(self, X):
        X = np.asarray(X)
        return self.density.pdf(X)

    def sample(self, n_samples=1):
        samples = self.density.rvs(size=n_samples,random_state=self.random_state)
        if n_samples == 1:
            samples = samples.reshape(1, -1)
        return samples

class UnsupervisedEnsemble(torch.nn.Module):
    def __init__(self, features_dim, beta = 1, alpha_rec = 0.1, alpha_sim = 1, alpha_div = 1, random_state = 42):
        super(UnsupervisedEnsemble, self).__init__()

        self.name = "UnsupervisedEnsemble"
        self.features_dim = features_dim
        self.beta = beta
        self.alpha_rec = alpha_rec
        self.alpha_sim = alpha_sim
        self.alpha_div = alpha_div
        self.random_state = random_state
        # Encoder
        self.fc1 = torch.nn.Linear(features_dim, 36)
        self.fc21 = torch.nn.Linear(36, 24)
        self.fc22 = torch.nn.Linear(36, 24)
        # Decoder
        self.fc3 = torch.nn.Linear(24, 36)
        self.fc4 = torch.nn.Linear(36, features_dim)

        self.experts = {}

    def encoder(self, x):
        h = torch.sigmoid(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def decoder(self, z):
        h = torch.sigmoid(self.fc3(z))
        return self.fc4(h)

    def vae_sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.vae_sample(mu, log_var)
        return self.decoder(z), mu, log_var

    def generated_dataset(self):
        samples = []

        for (cluster_id, submodel_name), expert in self.experts.items():
            density = expert["density"]
            samples.append(density.sample(1)[0])

        return torch.tensor(np.stack(samples)).float()

    def assign_submodels(self, predictors_info):
        for (cluster_id, submodel_name), info in predictors_info.items():
            density = DataDensity(
                info["Feature_mean"],
                info["Feature_var"],
                self.random_state,
            )

            self.experts[(cluster_id, submodel_name)] = {
                "density": density,
                "predictor": info["Submodel"],
            }

    def get_submodel_predictions(self, dataset):
        """
        dataset: torch.Tensor, shape (n_samples, n_features)
        """

        n_experts = len(self.experts)
        n_samples = dataset.shape[0]

        model_preds = torch.zeros(n_experts, n_samples)

        for i, expert in enumerate(self.experts.values()):
            submodel = expert["predictor"]

            if isinstance(submodel, torch.nn.Module):
                submodel.eval()
                with torch.no_grad():
                    pred = submodel(dataset).squeeze()

            else:
                pred = torch.from_numpy(
                    submodel.predict(dataset.detach().numpy())
                ).float()

            model_preds[i] = pred

        return model_preds

    def kl_loss(self, mu, log_var):
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def rec_loss(self, dataset, dataset_rec, mu, log_var):
        return ((dataset - dataset_rec) ** 2).mean() + self.beta * self.kl_loss(mu, log_var)

    def sim_loss(self, submodel_predictions, pairwise_distances, eps=1e-8):
        """
        submodel_predictions: (n_expert, n_sample)
        pairwise_distances:   (n_sample, n_sample)
        """

        # (n_sample, n_expert)
        preds = submodel_predictions.T

        # (n_sample, n_expert, n_expert)
        model_dist = torch.cdist(
            preds.unsqueeze(-1),
            preds.unsqueeze(-1),
            p=2
        )

        N = model_dist.shape[0]
        idx = torch.arange(N, device=model_dist.device)
        aligned_dist = model_dist[idx, idx, :]

        # normalization
        row_min = aligned_dist.min(dim=1, keepdim=True).values
        row_max = aligned_dist.max(dim=1, keepdim=True).values

        aligned_dist_norm = (aligned_dist - row_min) / (row_max - row_min + eps)

        # similarity
        model_similarity = 1.0 - aligned_dist_norm

        return (model_similarity * pairwise_distances).mean()

    def div_loss(self, pairwise_distances):
        return -0.5 * pairwise_distances.mean()

    def loss(self, pretraining_features):
        pretraining_features_recon, pretraining_mu, pretraining_log_var = self.forward(pretraining_features,)
        generated_features = self.generated_dataset()
        generated_features_recon, generated_mu, generated_log_var = self.forward(generated_features)
        pairwise_distances = torch.cdist(generated_mu, generated_mu)
        generated_mu_predictions = self.get_submodel_predictions(generated_features)

        combined_features_recon = torch.cat((pretraining_features_recon, generated_features_recon), dim=0)
        combined_features = torch.cat((pretraining_features, generated_features), dim=0)
        combined_mu = torch.cat((pretraining_mu, generated_mu), dim=0)
        combined_log_var = torch.cat((pretraining_log_var, generated_log_var), dim=0)
        return (self.alpha_rec * self.rec_loss(combined_features, combined_features_recon, combined_mu, combined_log_var)
               + self.alpha_sim * self.sim_loss(generated_mu_predictions, pairwise_distances)
               + self.alpha_div * self.div_loss(pairwise_distances))

    def fit(self, pretraining_features, optimizer, epochs = 10, unsupervised_ensemble_batch_size = 256,):

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        g = torch.Generator()
        g.manual_seed(self.random_state)
        optimizer = optimizer
        pretraining_features_loader = torch.utils.data.DataLoader(pretraining_features,
                                                                  batch_size = unsupervised_ensemble_batch_size,
                                                                  shuffle = True,
                                                                  drop_last = False,
                                                                  generator = g)
        for epoch in range(epochs):
            self.train()
            running_loss = 0
            for i, batch in enumerate(pretraining_features_loader):

                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()
                running_loss += loss
            average_loss = round((running_loss.detach().numpy() / (i + 1)), 5)

            print(f"Epoch {epoch+1} average loss: {average_loss}")

        return

    def remodel_in_z(self, n_samples = 1000, n_components = 3):

        for (cluster_id, submodel_name), expert in self.experts.items():
            density = expert["density"]
            with torch.no_grad():
                _, remodel_mu, _ = self.forward(torch.tensor(density.sample(n_samples), dtype=torch.float32))
                remodel_mu = remodel_mu.numpy()
            gmm = GaussianMixture(
                n_components= n_components,
                covariance_type = "full",
                random_state = self.random_state,
            )
            gmm.fit(remodel_mu)
            self.experts[(cluster_id, submodel_name)]["density_in_z"] = gmm

        self.rep_flag = True

    def get_weights_in_z(self, X, epsilon):
        if not self.rep_flag:
            raise RuntimeError(
                "Please call `remodel_in_z()` before `get_weights_in_z()`."
            )
        n_samples = X.shape[0]
        n_experts = len(self.experts)
        log_weights = torch.zeros(n_samples, n_experts, dtype = torch.float32)

        _, z_mu, _ = self.forward(X)

        for i,expert in enumerate(self.experts.values()):
            log_density = torch.from_numpy(expert["density_in_z"].score_samples(z_mu.detach().cpu().numpy())).float()
            log_weights[:, i] = log_density

        weights = torch.softmax(log_weights, dim=1)
        weights = (weights + epsilon)
        weights = weights / weights.sum(dim=1, keepdim=True)

        return weights

    def predict(self, X, epsilon=1e-10):

        weights = self.get_weights_in_z(X, epsilon)
        submodel_predictions = self.get_submodel_predictions(X).T
        assert submodel_predictions.shape == weights.shape, (
            f"Shape mismatch: submodel_predictions {submodel_predictions.shape}, "
            f"weights {weights.shape}"
        )

        predictions = (submodel_predictions * weights).sum(axis=1)

        return submodel_predictions, predictions, weights

    def save_model(self):

        ensemble_path = f"./Cash/{self.name}.pth"
        torch.save(self.state_dict(), ensemble_path)
        expert_order = []
        for (cluster_id, submodel_name), expert in self.experts.items():
            expert_order.append((cluster_id, submodel_name))
            submodel = expert["predictor"]
            submodel_path = f"./Cash/submodel_{submodel_name}_{cluster_id}.pkl"
            z_density_path = f"./Cash/z_density_{submodel_name}_{cluster_id}.pkl"
            if isinstance(submodel, torch.nn.Module):
                torch.save(submodel.state_dict(), submodel_path.replace(".pkl", ".pth"))
            else:
                joblib.dump(submodel, submodel_path)
            z_density = expert["density_in_z"]
            joblib.dump(z_density, z_density_path)
        meta = {"expert_order": expert_order,}
        joblib.dump(meta, f"./Cash/meta.pkl")

    def load_model(self, nn_factory):

        ensemble_path = f"./Cash/{self.name}.pth"
        self.load_state_dict(torch.load(ensemble_path))
        meta = joblib.load("./Cash/meta.pkl")
        expert_order = meta["expert_order"]

        self.experts = {}

        for cluster_id, submodel_name in expert_order:

            base = f"./Cash/submodel_{submodel_name}_{cluster_id}"
            pkl_path = base + ".pkl"
            pth_path = base + ".pth"
            z_density_path = f"./Cash/z_density_{submodel_name}_{cluster_id}.pkl"

            if os.path.exists(pth_path):
                if submodel_name not in nn_factory:
                    raise ValueError(
                        f"NN architecture for '{submodel_name}' not provided in nn_factory"
                    )
                submodel = nn_factory[submodel_name]()
                submodel.load_state_dict(torch.load(pth_path))
                submodel.eval()
            elif os.path.exists(pkl_path):
                submodel = joblib.load(pkl_path)
            else:
                raise FileNotFoundError(
                    f"No submodel file found for {submodel_name}, cluster {cluster_id}"
                )

            density_in_z = joblib.load(z_density_path)

            self.experts[(cluster_id, submodel_name)] = {
                "predictor": submodel,
                "density_in_z": density_in_z,
            }

import torch
import numpy as np
from Utils.Args import all_args
from Utils.Preprocessing import prepare_data
from model.SubmodelTraining import run_causal_forest_and_clustering,run_all_submodels_and_collect_results,\
    submodel_registry,hierarchical_submodel_allocation,submodel_factory
from model.Ensemble import UnsupervisedEnsemble
from model.Calibration import Gate
from Utils.EvaluationMetrics import RMSE,MAE,IA,R2


def main():

    args = all_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    df_pretraining, df_calibration, df_evaluation = prepare_data(args)
    clustered_data = run_causal_forest_and_clustering(df_pretraining, args, )
    submodel_grid_search_results = run_all_submodels_and_collect_results(clustered_data, submodel_registry, args, )
    predictors_info = hierarchical_submodel_allocation(submodel_grid_search_results, submodel_factory, clustered_data,
                                                       args, )

    ensemble = UnsupervisedEnsemble(len(args.features), beta = args.beta, alpha_rec = args.alpha_rec,
                                    alpha_sim = args.alpha_sim, alpha_div = args.alpha_div, random_state = args.random_state)
    ensemble.assign_submodels(predictors_info)
    ensemble_optimizer = torch.optim.Adam(ensemble.parameters(), lr = 0.01, betas = (0.9, 0.9))
    ensemble.fit(torch.tensor(df_pretraining[args.features].values, dtype = torch.float32),
                 ensemble_optimizer,
                 epochs = args.ensemble_epochs,
                 unsupervised_ensemble_batch_size = args.ensemble_batch_size)
    ensemble.eval()
    ensemble.remodel_in_z(n_samples = args.n_samples)


    gate = Gate(len(args.features), ensemble, args.lambda_fusion, args.lambda_rank,
                 args.delta, args.xi, args.Temperature, args.random_state, args.use_xavier_init)
    gate.train()
    gate_optimizer = torch.optim.Adam(gate.parameters(), lr = 0.01, betas = (0.9, 0.9))
    gate.fit(torch.tensor(df_calibration[args.features].values, dtype = torch.float32),
             torch.tensor(df_calibration[args.target].values, dtype = torch.float32),
             optimizer = gate_optimizer,
             epochs = args.gate_epochs,
             batch_size = args.gate_batch_size,
             )
    gate.eval()
    _, final_predictions, _ = gate(torch.tensor(df_evaluation[args.features].values, dtype=torch.float32))
    print("RMSE:", RMSE(df_evaluation[args.target].values, final_predictions))
    print("MAE :", MAE(df_evaluation[args.target].values, final_predictions))
    print("R2:", R2(df_evaluation[args.target].values, final_predictions))
    print("IA  :", IA(df_evaluation[args.target].values, final_predictions))

if __name__ == "__main__":
    main()



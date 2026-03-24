import argparse
import json

def all_args():
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for causal forest models")

    parser.add_argument("--target", type=str, default="pifc", choices=["pifc", "pcfc"])
    parser.add_argument("--require_last_target", action="store_true",
                        help="Whether to require last_{target} to be non-missing")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretraining_end", type=str, default="2021-08-04")
    parser.add_argument("--evaluation_start", type=str, default="2021-08-11")
    parser.add_argument("--knn_neighbors", type=int, default=10)
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        default=[
            "last_pifc",
            "gdp_adm0",
            "ce_variation_3months",
            "headline_inflation_value",
            "single_pewi_max_last_3months",
            "food_inflation_value",
            "rainfall_value_mean_last_12month",
            "rainfall_1_month_anomaly_max_last_3month",
            "ndvi_value_mean_last_12month",
            "ndvi_anomaly_mean_last_3month",
            "num_fatalities_battles_remote_violence_90days_difference",
            "prevalence_of_undernourishment",
            "pop_density",
        ],
        help="List of feature column names"
    )
    parser.add_argument(
        "--feature_clip",
        nargs="+",
        type=str,
        default=[
            "gdp_adm0:clip,0,0.95",
            "ce_variation_3months:clip,0.05,0.9",
            "headline_inflation_value:clip,0.05,0.95",
            "single_pewi_max_last_3months:clip,0.05,0.95",
            "food_inflation_value:clip,0.05,0.95",
            "rainfall_value_mean_last_12month:clip,0,0.99",
            "rainfall_1_month_anomaly_max_last_3month:clip,0.01,0.9",
            "ndvi_value_mean_last_12month:noclip",
            "ndvi_anomaly_mean_last_3month:clip,0.05,0.9",
            "num_fatalities_battles_remote_violence_90days_difference:clip,0.05,0.95",
            "prevalence_of_undernourishment:noclip",
            "pop_density:clip,0,0.9",
        ],
        help=(
            "Feature clipping config. "
            "Format: feature_name:clip,min,max or feature_name:noclip"
        )
    )
    parser.add_argument(
        "--no_scale_features",
        nargs="+",
        type=str,
        default=[
            "last_pifc",
            "ndvi_value_mean_last_12month",
        ],
        help="Feature column names that should NOT be scaled"
    )

    parser.add_argument("--causal_forest_n_estimators", type=int, default=100, help="Number of trees in causal forest")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for KMeans")
    parser.add_argument("--save_clusters", action="store_true", help="Whether to save clustered data to CSV files")

    parser.add_argument(
        "--submodel_configs",
        type=json.loads,
        default={
            "AdaBoost": {
                "n_estimators": [50, 60, 70, 80, 90, 100],
                "learning_rate": [0.02, 0.04, 0.06, 0.08, 0.1],
            },
            "DT": {
                "max_depth": [2, 3, 4, 5, 6],
                "min_samples_split": [10, 20, 50, 100],
                "min_samples_leaf": [5, 10, 20, 30],
            },
            "CatBoost": {
                "depth": [2, 3, 4, 5],
                "iterations": [50, 60, 70, 80],
                "learning_rate": [0.02, 0.04, 0.06, 0.08, 0.1],
            },
            "DNN": {
                "model_name": ["bpnn"],
                "loss_name": ["mse"],
                "optimizer_name": ["adam"],
                "batch_size": [256],
                "epochs": [100],
            },
            "ET": {
                "n_estimators": [50, 60, 70, 80, 90, 100],
                "max_depth": [4, 5, 6, 7, 8],
                "min_samples_split": [10, 20, 50, 100],
            },
            "GBDT": {
                "max_depth": [2, 3, 4, 5],
                "n_estimators": [30, 35, 40, 45, 50],
                "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            },
            "KNNR": {
                "n_neighbors": [50, 60, 70],
            },
            "LASSO": {
                "alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            },
            "LightGBM": {
                "max_depth": [2, 3, 4, 5],
                "n_estimators": [30, 35, 40, 45, 50],
                "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            },
            "LR": {
            },
            "RF": {
                "n_estimators": [50, 60, 70, 80, 90, 100],
                "max_depth": [2, 3, 4, 5, 6],
                "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            },
            "Ridge": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "SVR": {
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1, 0.2, 0.5],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            },
            "XGBoost": {
                "max_depth": [2, 3, 4, 5],
                "n_estimators": [30, 35, 40, 45, 50],
                "learning_rate": [0.02, 0.04, 0.06, 0.08],
            },
        },
        help="JSON string defining models and their hyperparameters."
    )
    parser.add_argument("--save_grid_search", action="store_true",
                        help="Whether to save submodel grid search results to CSV files")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top submodels selected per cluster")

    parser.add_argument("--beta", type=float, default=1, help="The regularization coefficient of the reconstruction loss")
    parser.add_argument("--alpha_rec", type=float, default=1, help="Weight for the reconstruction loss")
    parser.add_argument("--alpha_sim", type=float, default=1, help="Weight for the similarity loss")
    parser.add_argument("--alpha_div", type=float, default=0.1, help="Weight for the diversity loss")
    parser.add_argument("--ensemble_epochs", type=int, default=10, help="Number of epochs for unsupervised ensemble training")
    parser.add_argument("--ensemble_batch_size", type=int, default=256,
                        help="Batch size for unsupervised ensemble training")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")

    parser.add_argument("--lambda_fusion", type=float, default=1, help="Weight for fusion loss")
    parser.add_argument("--lambda_rank", type=float, default=0.1, help="Weight for rank loss")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="A predefined threshold that sets the maximum allowable range for the weights of less accurate predictions to exceed the weights of the best-performing predictions.")
    parser.add_argument("--xi", type=float, default=1.0, help="A hyperparameter that controls the strength of the ranking constraint")
    parser.add_argument("--Temperature", type=float, default=5, help="The temperature hyperparameter used to control the smoothness of the ensemeble weights")
    parser.add_argument("--use_xavier_init", type=bool, default=False,
                        help="Whether to use Xavier initialization for the update and reset gates")
    parser.add_argument("--gate_epochs", type=int, default=10, help="Number of epochs for gate model training")
    parser.add_argument("--gate_batch_size", type=int, default=256, help="Batch size for gate ensemble")

    return parser.parse_args()
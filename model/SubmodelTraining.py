from Utils.Args import all_args
from Utils.Preprocessing import prepare_data
from model.DNNSubmodel import train_nn_model
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
import itertools
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import catboost
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import xgboost
import ast


def run_causal_forest_and_clustering(df_pretraining: pd.DataFrame, args,):

    target = args.target
    features = args.features
    causal_effects_dict = {}
    Y = df_pretraining[target]

    for feature in features:
        T = df_pretraining[feature]
        X = df_pretraining[features].drop(columns=[feature])
        causal_forest = CausalForestDML(
            model_t=RandomForestRegressor(random_state=args.random_state),
            model_y=RandomForestRegressor(random_state=args.random_state),
            discrete_treatment=False,
            n_estimators=args.causal_forest_n_estimators,
            random_state=args.random_state,
        )
        causal_forest.fit(Y=Y, T=T, X=X)
        treatment_effects = causal_forest.effect(X)
        causal_effects_dict[feature] = treatment_effects

    causal_df = pd.DataFrame(causal_effects_dict)
    causal_df = causal_df.add_prefix("causal_effect_")
    cluster_columns = list(causal_df.columns)
    merged_df = pd.concat([df_pretraining.reset_index(drop=True), causal_df], axis=1,)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state,)
    clusters = kmeans.fit_predict(merged_df[cluster_columns])

    df_pretraining_with_clusters = df_pretraining.copy()
    df_pretraining_with_clusters["Cluster"] = clusters
    clustered_data = {}
    for cluster_id in range(args.n_clusters):
        cluster_df = df_pretraining_with_clusters[df_pretraining_with_clusters["Cluster"] == cluster_id]
        clustered_data[cluster_id] = cluster_df
        if args.save_clusters:
            filename = f"./Cash/cluster_{cluster_id}.csv"
            cluster_df.to_csv(filename)

    return clustered_data

submodel_registry = {}
def register_submodel(name):
    def decorator(func):
        submodel_registry[name] = func
        return func
    return decorator

@register_submodel("AdaBoost")
def adaboost_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "AdaBoost"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = AdaBoostRegressor(
                estimator = DecisionTreeRegressor(random_state = args.random_state),
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("DT")
def dt_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "DT"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = DecisionTreeRegressor(
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("CatBoost")
def catboost_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "CatBoost"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = catboost.CatBoostRegressor(
                loss_function = 'RMSE',
                verbose = 0,
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("DNN")
def dnn_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "DNN"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = train_nn_model(
                cluster_data = cluster_data,
                args = args,
                **params_dict,
            )
            submodel.eval()
            preds = submodel(torch.tensor(X, dtype = torch.float32)).detach().numpy()
            rmse = np.sqrt(mean_squared_error(y, preds.squeeze()))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("ET")
def et_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "ET"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = ExtraTreesRegressor(
                n_jobs = -1,
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("GBDT")
def gbdt_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "GBDT"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = GradientBoostingRegressor(
                loss = 'squared_error',
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("KNNR")
def knnr_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "KNNR"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = KNeighborsRegressor(
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("LASSO")
def lasso_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "LASSO"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = Lasso(
                max_iter = 1000,
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("LightGBM")
def lightgbm_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "LightGBM"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = lgb.LGBMRegressor(
                objective = 'regression',
                metric = 'rmse',
                verbosity = -1,
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("LR")
def lr_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "LR"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = LinearRegression(
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("RF")
def rf_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "RF"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = RandomForestRegressor(
                n_jobs = -1,
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("Ridge")
def ridge_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "Ridge"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = Ridge(
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("SVR")
def svr_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "SVR"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = SVR(
                kernel="rbf",
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

@register_submodel("XGBoost")
def xgboost_grid_search_by_cluster(clustered_data: dict, args):
    submodel_name = "XGBoost"
    param_grid = args.submodel_configs.get(submodel_name, {})
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    cluster_results = {}
    for cluster_id, cluster_data in clustered_data.items():
        best_rmse = float("inf")
        best_params = None
        X = cluster_data[args.features].to_numpy()
        y = cluster_data[args.target].to_numpy()
        if param_names:
            param_combinations = itertools.product(*param_values)
        else:
            param_combinations = [()]

        for combo in param_combinations:
            params_dict = dict(zip(param_names, combo))
            submodel = xgboost.XGBRegressor(
                objective = "reg:logistic",
                random_state = args.random_state,
                **params_dict,
            )
            submodel.fit(X, y)
            preds = submodel.predict(X)
            rmse = np.sqrt(mean_squared_error(y, preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params_dict

        cluster_results[cluster_id] = {
            "cluster_id": cluster_id,
            "best_params": best_params,
            "best_rmse": best_rmse,
        }
    return cluster_results

def run_all_submodels_and_collect_results(
    clustered_data: dict,
    submodel_registry: dict,
    args,
) -> pd.DataFrame:
    """
    Run all registered models with provided hyperparameters and
    return a combined results DataFrame.
    """

    all_results = {}
    for submodel_name in list(args.submodel_configs.keys()):
        if submodel_name not in submodel_registry:
            raise ValueError(
                f"Submodel '{submodel_name}' is not registered. "
                f"Available models: {list(submodel_registry.keys())}"
            )

        print(f"\n===== Running {submodel_name} =====")
        result = submodel_registry[submodel_name](
            clustered_data = clustered_data,
            args = args,
        )
        all_results[submodel_name] = result

    rows = []
    for submodel_name, results in all_results.items():
        for cluster_id, cluster_result in results.items():
            rows.append({
                "Submodel": submodel_name,
                "Cluster ID": cluster_result["cluster_id"],
                "Best RMSE": cluster_result["best_rmse"],
                "Best Params": cluster_result["best_params"],
            })
    submodel_grid_search_results = pd.DataFrame(rows)
    if args.save_grid_search:
        filename = f"./Cash/submodel_grid_search_results.csv"
        submodel_grid_search_results.to_csv(filename, index=False)
    return submodel_grid_search_results

submodel_factory = {}
def register_submodel_builder(name):
    def decorator(func):
        submodel_factory[name] = func
        return func
    return decorator

@register_submodel_builder("AdaBoost")
def build_adaboost(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = AdaBoostRegressor(
               estimator = DecisionTreeRegressor(random_state = args.random_state),
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("DT")
def build_dt(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = DecisionTreeRegressor(
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("CatBoost")
def build_catboost(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = catboost.CatBoostRegressor(
               loss_function = 'RMSE',
               verbose = 0,
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("DNN")
def build_dnn(cluster_data, params_dict, args, **kwargs):
    submodel = train_nn_model(
               cluster_data = cluster_data,
               args = args,
               **params_dict,
            )
    return submodel
@register_submodel_builder("ET")
def build_et(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = ExtraTreesRegressor(
               n_jobs = -1,
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("GBDT")
def build_gbdt(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = GradientBoostingRegressor(
               loss = 'squared_error',
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("KNNR")
def build_knnr(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = KNeighborsRegressor(
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("LASSO")
def build_lasso(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = Lasso(
               max_iter = 1000,
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("LightGBM")
def build_lightgbm(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = lgb.LGBMRegressor(
               objective = 'regression',
               metric = 'rmse',
               verbosity = -1,
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("LR")
def build_lr(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = LinearRegression(
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("RF")
def build_rf(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = RandomForestRegressor(
               n_jobs = -1,
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("Ridge")
def build_ridge(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = Ridge(
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("SVR")
def build_svr(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = SVR(
               kernel="rbf",
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel
@register_submodel_builder("XGBoost")
def build_xgboost(cluster_data, params_dict, args, **kwargs):
    X = cluster_data[args.features].to_numpy()
    y = cluster_data[args.target].to_numpy()
    submodel = xgboost.XGBRegressor(
               objective = "reg:logistic",
               random_state = args.random_state,
               **params_dict,
            )
    submodel.fit(X,y)
    return submodel

def safe_parse_params(x):
    if isinstance(x, dict):
        return x
    if pd.isna(x) or x == "":
        return {}
    return ast.literal_eval(x)

def hierarchical_submodel_allocation(
    submodel_grid_search_results: pd.DataFrame,
    submodel_factory: dict,
    clustered_data: dict,
    args,
):
    """
    Select top-k submodels per cluster (by lowest RMSE),
    ensure each submodel is used at least once,
    then train submodels and return predictors.

    Returns
    -------
    predictors : dict
        {(cluster_id, submodel_name): {
            "Submodel": trained_submodel,
            "Feature_mean": Series,
            "Feature_var": Series,
        }}
    """
    allocation_strategy = []
    submodel_name_set = set(submodel_grid_search_results["Submodel"].unique())
    cluster_ids = submodel_grid_search_results["Cluster ID"].unique()

    for cluster_id in cluster_ids:
        cluster_df = submodel_grid_search_results[
            submodel_grid_search_results["Cluster ID"] == cluster_id
        ]

        topk_submodels = (
            cluster_df.sort_values("Best RMSE").head(args.top_k)["Submodel"].tolist()
        )
        for submodel in topk_submodels:
            allocation_strategy.append((int(cluster_id), submodel))

    used_submodels = {submodel for _, submodel in allocation_strategy}
    unused_submodels = submodel_name_set - used_submodels
    for submodel in unused_submodels:
        best_cluster = None
        min_error = float("inf")
        for cluster_id in cluster_ids:
            row = submodel_grid_search_results[
                (submodel_grid_search_results["Cluster ID"] == cluster_id) &
                (submodel_grid_search_results["Submodel"] == submodel)
            ]
            if not row.empty:
                error = row["Best RMSE"].iloc[0]
                if error < min_error:
                    min_error = error
                    best_cluster = cluster_id
        if best_cluster is not None:
            allocation_strategy.append((int(best_cluster), submodel))
    predictors_info = {}
    for cluster_id, submodel_name in allocation_strategy:
        cluster_data = clustered_data[cluster_id]
        submodel_params = safe_parse_params(submodel_grid_search_results[
            (submodel_grid_search_results['Submodel'] == submodel_name)
            & (submodel_grid_search_results['Cluster ID'] == cluster_id)]['Best Params'].values[0])
        submodel = submodel_factory[submodel_name](
            cluster_data = cluster_data,
            params_dict = submodel_params,
            args = args,
        )
        predictors_info[(cluster_id, submodel_name)] = {
            "Submodel": submodel,
            "Feature_mean": cluster_data[args.features].mean(),
            "Feature_var": cluster_data[args.features].var(),
        }
    print(f"Submodels Training Successfully!")
    return predictors_info
if __name__ == "__main__":
    args = all_args()
    df_pretraining, df_calibration, df_evaluation = prepare_data(args)
    clustered_data = run_causal_forest_and_clustering(df_pretraining, args,)
    submodel_grid_search_results = run_all_submodels_and_collect_results(clustered_data, submodel_registry, args,)
    predictors_info = hierarchical_submodel_allocation(submodel_grid_search_results, submodel_factory, clustered_data, args,)
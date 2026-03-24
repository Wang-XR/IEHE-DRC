import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from Utils.Args import all_args


def parse_feature_clip_config(feature_clip_list):
    config = {}
    for item in feature_clip_list:
        name, rule = item.split(":", 1)

        if rule == "noclip":
            config[name] = {"mode": "noclip"}
        else:
            parts = rule.split(",")
            if parts[0] != "clip" or len(parts) != 3:
                raise ValueError(f"Invalid feature_clip rule: {item}")

            qmin = float(parts[1])
            qmax = float(parts[2])

            if not (0 <= qmin < qmax <= 1):
                raise ValueError(f"Invalid quantile range in {item}")

            config[name] = {
                "mode": "clip",
                "qmin": qmin,
                "qmax": qmax,
            }
    return config


def compute_clip_bounds(df, clip_config):
    bounds = {}
    for col, rule in clip_config.items():
        if rule["mode"] == "clip" and col in df.columns:
            lower = df[col].quantile(rule["qmin"])
            upper = df[col].quantile(rule["qmax"])
            bounds[col] = (lower, upper)
    return bounds


def apply_clip(df, clip_bounds):
    df = df.copy()
    for col, (lower, upper) in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)
    return df


def prepare_data(args):
    target = args.target
    data_dir = args.data_dir
    features = args.features

    df = pd.read_csv(os.path.join(data_dir, "Food_dataset.csv"))
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df[~df[target].isna()]
    df.reset_index(drop=True, inplace=True)
    selected_columns = features + [target]
    if args.require_last_target:
        df = df.loc[~df[f"last_{target}"].isna()].reset_index(drop=True)
    else:
        df = df.loc[df[f"last_{target}"].isna()].reset_index(drop=True)
    pretraining_end = pd.to_datetime(args.pretraining_end).date()
    evaluation_start = pd.to_datetime(args.evaluation_start).date()
    df_pretraining = df[df["date"].dt.date <= pretraining_end][selected_columns]
    df_calibration = df[
        (df["date"].dt.date > pretraining_end) &
        (df["date"].dt.date < evaluation_start)
        ][selected_columns]
    df_evaluation = df[df["date"].dt.date >= evaluation_start][selected_columns]

    clip_config = parse_feature_clip_config(args.feature_clip)
    clip_bounds = compute_clip_bounds(df_pretraining, clip_config)
    df_pretraining = apply_clip(df_pretraining, clip_bounds)
    df_calibration = apply_clip(df_calibration, clip_bounds)
    df_evaluation = apply_clip(df_evaluation, clip_bounds)

    no_scale_features = args.no_scale_features + [target]
    scale_features = [f for f in features if f not in no_scale_features]
    scaler = MinMaxScaler()
    df_uncsaler_pretraining = df_pretraining[no_scale_features]
    df_csaler_pretraining = pd.DataFrame(
        scaler.fit_transform(df_pretraining[scale_features]),
        columns=scale_features,
        index=df_pretraining.index
    )
    df_pretraining = pd.concat([df_uncsaler_pretraining, df_csaler_pretraining], axis=1)
    df_uncsaler_calibration = df_calibration[no_scale_features]
    df_csaler_calibration = pd.DataFrame(
        scaler.transform(df_calibration[scale_features]),
        columns=scale_features,
        index=df_calibration.index
    )
    df_calibration = pd.concat([df_uncsaler_calibration, df_csaler_calibration], axis=1)
    df_uncsaler_evaluation = df_evaluation[no_scale_features]
    df_csaler_evaluation = pd.DataFrame(
        scaler.transform(df_evaluation[scale_features]),
        columns=scale_features,
        index=df_evaluation.index
    )
    df_evaluation = pd.concat([df_uncsaler_evaluation, df_csaler_evaluation], axis=1)

    imputer = KNNImputer(n_neighbors=args.knn_neighbors)
    target_pretraining = df_pretraining[target]
    features_pretraining = pd.DataFrame(
        imputer.fit_transform(df_pretraining[features]),
        columns=features,
        index=df_pretraining.index
    )
    df_pretraining = pd.concat([features_pretraining, target_pretraining], axis=1)
    target_calibration = df_calibration[target]
    features_calibration = pd.DataFrame(
        imputer.transform(df_calibration[features]),
        columns=features,
        index=df_calibration.index
    )
    df_calibration = pd.concat([features_calibration, target_calibration], axis=1)
    target_evaluation = df_evaluation[target]
    features_evaluation = pd.DataFrame(
        imputer.transform(df_evaluation[features]),
        columns=features,
        index=df_evaluation.index
    )
    df_evaluation = pd.concat([features_evaluation, target_evaluation], axis=1)

    return df_pretraining, df_calibration, df_evaluation
if __name__ == "__main__":
    args = all_args()
    df_pretraining, df_calibration, df_evaluation = prepare_data(args)
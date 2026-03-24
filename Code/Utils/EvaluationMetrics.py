import numpy as np
import torch

def RMSE(actual, predict):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    actual = np.asarray(actual)
    predict = np.asarray(predict)

    return np.sqrt(np.mean((actual - predict) ** 2))

def MAE(actual, predict):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    actual = np.asarray(actual)
    predict = np.asarray(predict)

    return np.mean(np.abs(actual - predict))

def R2(actual, predict, eps=1e-8):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    actual = np.asarray(actual)
    predict = np.asarray(predict)

    ss_res = np.sum((actual - predict) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)

    return 1 - ss_res / (ss_tot + eps)

def IA(actual, predict, eps=1e-8):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()

    actual = np.asarray(actual)
    predict = np.asarray(predict)

    mean_actual = actual.mean()
    num = np.sum((actual - predict) ** 2)
    den = np.sum((np.abs(predict - mean_actual) + np.abs(actual - mean_actual)) ** 2)

    return 1 - num / (den + eps)



import numpy as np

def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    Avoids division by zero and handles zero values safely.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    nonzero = denominator != 0
    smape_values = np.zeros_like(denominator)
    smape_values[nonzero] = np.abs(y_true[nonzero] - y_pred[nonzero]) / denominator[nonzero]
    return 100.0 * np.mean(smape_values)

def evaluate_global_metrics(y_true, y_pred):
    """
    Returns a dictionary containing MAE, RMSE, and sMAPE.
    Useful for standardized performance comparison across models.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred)
    }

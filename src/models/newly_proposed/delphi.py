import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import timedelta

# Logistic transmission reduction due to interventions
def intervention_effect(t, r, k, t0):
    return 1.0 - r / (1 + np.exp(-k * (t - t0)))

def seird_model_with_policy(t, y, alpha, beta0, gamma, delta, r, k, t0):
    S, E, I, R, D = y
    N = S + E + I + R + D
    beta_t = beta0 * intervention_effect(t, r, k, t0)
    dSdt = -beta_t * S * I / N
    dEdt = beta_t * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def fit_seird_model(t_data, cases, deaths, population, config):
    I0 = max(cases[0], 1)
    E0 = I0 * 2
    R0 = 0
    D0 = deaths[0]
    S0 = population - E0 - I0 - R0 - D0
    y0 = [S0, E0, I0, R0, D0]
    t_span = (t_data[0], t_data[-1])
    t_eval = np.arange(t_data[0], t_data[-1] + 1)

    def loss(params):
        alpha, beta0, gamma, delta, r, k, t0 = params
        try:
            sol = solve_ivp(
                lambda t, y: seird_model_with_policy(t, y, alpha, beta0, gamma, delta, r, k, t0),
                t_span, y0, t_eval=t_eval
            )
            I_pred = sol.y[2]
            D_pred = sol.y[4]
            return np.mean((I_pred - cases) ** 2) + np.mean((D_pred - deaths) ** 2)
        except:
            return np.inf

    p0 = config.get('p0', [0.2, 0.5, 0.1, 0.01, 0.5, 0.1, t_data[len(t_data)//2]])
    bounds = config.get('bounds', [
        (0.01, 1.0),
        (0.01, 2.0),
        (0.01, 1.0),
        (0.001, 0.1),
        (0.0, 1.0),
        (0.01, 1.0),
        (t_data[0], t_data[-1])
    ])

    result = minimize(loss, p0, bounds=bounds, method='L-BFGS-B')
    return result, y0

def predict_seird(t_start, n_days, y0, fitted_params):
    t_eval = np.arange(t_start, t_start + n_days + 1)
    sol = solve_ivp(
        lambda t, y: seird_model_with_policy(t, y, *fitted_params),
        (t_eval[0], t_eval[-1]), y0, t_eval=t_eval
    )
    return sol.t, sol.y

def load_country_data(file_path, population_path, country):
    df = pd.read_csv(file_path, parse_dates=['date'], index_col=0)
    pop_df = pd.read_csv(population_path)
    country_df = df[df['country'] == country].copy()
    country_df.sort_values('date', inplace=True)

    country_df['new_cases_smoothed'] = country_df['new_cases'].rolling(window=7, min_periods=1).mean()
    country_df['new_deaths_smoothed'] = country_df['new_deaths'].rolling(window=7, min_periods=1).mean()

    population = pop_df[(pop_df['Country'] == country) & (pop_df['Province'].isna())]['pop2016'].values[0]
    return country_df.reset_index(), population

def fit_and_forecast(config):
    file_path = config['file_path']
    population_path = config['population_path']
    countries = config['countries'] if 'countries' in config else [config['country']]
    n_days_fit = config.get('n_days_fit', 30)
    n_days_forecast = config.get('n_days_forecast', 14)
    split_ratio = config.get('split_ratio', 0.7)

    all_results = []

    for country in countries:
        df, population = load_country_data(file_path, population_path, country)
        total_len = len(df)
        split_point = int(total_len * split_ratio)

        for i in range(split_point, total_len - n_days_forecast):
            start_date = df.loc[i - n_days_fit, 'date']
            fit_df = df[(df['date'] >= start_date) & (df['date'] <= df.loc[i, 'date'])].copy()
            cases = fit_df['new_cases_smoothed'].fillna(0).values
            deaths = fit_df['new_deaths_smoothed'].fillna(0).values
            t_data = np.arange(len(fit_df))

            try:
                result, y0 = fit_seird_model(t_data, cases, deaths, population, config)
                if not result.success:
                    continue
                t_forecast, y_forecast = predict_seird(len(t_data)-1, n_days_forecast, y0, result.x)

                forecast_origin = df.loc[i, 'date']
                forecast_dates = pd.date_range(start=forecast_origin + pd.Timedelta(days=1), periods=n_days_forecast)
                y_true = df['new_cases_smoothed'].iloc[i+1:i+1+n_days_forecast].fillna(0).values
                y_pred = y_forecast[2][1:]  # I values (infected), skip t0

                all_results.append({
                    "country": country,
                    "forecast_origin": forecast_origin,
                    "true_values": y_true.tolist(),
                    "predictions": y_pred.tolist(),
                    "forecast_dates": forecast_dates.tolist()
                })
            except:
                continue

    return pd.DataFrame(all_results)

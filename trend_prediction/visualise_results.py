import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import hiplot as hip


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def visualise_bohb(results=None, manual_config=None):
    results = pd.read_pickle(results)

    def transform_results(results, manual_config=None):
        transformed_results = defaultdict(list)
        # for run in results["test score"]:
        #     transformed_results["Slope RMSE"].append(run[0])
        #     transformed_results["Duration RMSE"].append(run[1])
        for column in results.columns:
            if column in ["n_epochs", "algorithm", "test std", "test score"]:
                continue
            elif column == "hidden_layer_dim" and "auto_mlp_regressor" in str(results["algorithm"]):
                for run in results[column]:
                    for layer in range(5):
                        try:
                            transformed_results[f"hidden_layer_{layer + 1}"].append(run[layer])
                        except IndexError:
                            transformed_results[f"hidden_layer_{layer + 1}"].append(np.NaN)
            elif column == "n_cells" and "bohb_lstm_regressor" in str(results["algorithm"]):
                for run in results[column]:
                    for layer in range(3):
                        try:
                            transformed_results[f"lstm_cells_{layer + 1}"].append(run[layer])
                        except IndexError:
                            transformed_results[f"lstm_cells_{layer + 1}"].append(np.NaN)
            else:
                transformed_results[column] = results[column]
        return transformed_results

    transformed_results = transform_results(results)
    transformed_results = pd.DataFrame(transformed_results)
    columns = [f"Run {run}" for run in transformed_results.index]
    transformed_results.index = columns
    transformed_results = transformed_results.T
    transformed_results["Manual tuning"] = manual_config
    hip.Experiment.from_iterable(transformed_results).display(force_full_width=True)
    # print(transformed_results)
    # transformed_results.replace(0, 1e-6, inplace=True)
    # axes = transformed_results.plot(logy=True, grid=True,
    #                                   solid_joinstyle="round",
    #                                   xticks=range(len(transformed_results.index)))
    # axes.set_xlabel("Hyperparameter")
    # axes.set_ylabel("Value found by BOHB (log scale)")
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 10})
    mlp_jse = "results/best_configs/" \
              "best_config_auto_mlp_regressor_jse_pointdata2020-01-02 01-18-09-198041.pkl"
    mlp_methane = "results/best_configs/" \
                  "best_config_auto_mlp_regressor_methane_pointdata2020-01-17 06-12-41-784925.pkl"
    mlp_nyse = "results/best_configs/" \
               "best_config_auto_mlp_regressor_nyse_pointdata2020-01-03 09-39-28-948971-fixed-run1-result.pkl"
    lstm_nyse = "results/" \
                "best_configs/best_config_bohb_lstm_regressor_nyse_pointdata2020-01-03 05-05-09-477322.pkl"
    lstm_methane = "results/best_configs/" \
                   "best_config_bohb_lstm_regressor_methane_pointdata2020-01-13 18-32-07-441277.pkl"
    mlp_jse_manual_params = [250, 0.0, 1e-3, 0.0, 100, np.NaN, np.NaN, np.NaN, np.NaN]
    mlp_methane_manual_params = [250, 0.0, 1e-3, 0.0, 500, 400, 300, np.NaN, np.NaN]
    mlp_nyse_manual_params = [5000, 0.0, 1e-3, 5e-4, 500, 400, 300, np.NaN, np.NaN]
    lstm_methane_manual_params = [2000, 0.0, 1e-4, 0.0, 600, 300, np.NaN]
    lstm_nyse_manual_params = [5000, 0.5, 1e-3, 5e-5, 100, np.NaN, np.NaN]
    visualise_bohb(results=lstm_nyse, manual_config=mlp_nyse_manual_params)
    import sys
    sys.exit(0)
    algorithms = ["MLP", "LSTM", "CNN", "RF", "GBM", "SVR"]
    # power = 100.0*(136.47 - np.array([83.78, 80.83, 82.57, 83.65, 80.64, 81.99]))/136.47
    # co2 = 100.0*(12.48 - np.array([7.41, 7.41, 7.41, 7.09, 6.29, 5.85]))/12.48
    # gradient_results = pd.DataFrame({"Power": power, "CO2": co2}, index=algorithms)

    voltage_slope = 100.0*(17.09 - np.array([9.04, 10.30, 9.24, 9.53, 10.00, 9.32, 9.25]))/17.09
    voltage_slope_rmse = np.array([9.04, 10.30, 9.24, 9.53, 10.00, 9.32, 9.25])
    voltage_slope_trend = 100.0*(17.09 - np.array([9.30, 9.32, 9.30, 9.43, 10.10, 9.42]))/17.09
    voltage_slope_trend_local_data = 100.0*(17.09 - np.array([9.03, 10.30, 33.26, 9.35, 10.01, 9.54]))/17.09
    voltage_slope = np.append(voltage_slope, [39.11])
    voltage_duration = 100.0*(86.51 - np.array([62.82, 62.87, 62.40, 63.11, 62.67, 62.58, 62.37]))/86.51
    voltage_duration_rmse = np.array([62.82, 62.87, 62.40, 63.11, 62.67, 62.58, 62.37])
    voltage_duration_trend = 100.0*(86.51 - np.array([62.72, 62.71, 62.61, 70.10, 62.63, 62.68]))/86.51
    voltage_duration_trend_local_data = 100.0*(86.51 - np.array([62.81, 62.87, 90.78, 63.19, 62.63, 62.62]))/86.51
    voltage_duration = np.append(voltage_duration, [35.43])

    methane_slope = 100.0*(28.54 - np.array([14.57, 14.21, 15.07, 10.09, 13.05, 14.98, 14.87]))/28.54
    methane_slope_rmse = np.array([14.57, 14.21, 15.07, 10.09, 13.05, 14.98, 14.87])
    methane_slope = np.append(methane_slope, [10.50])
    methane_duration = 100.0*(152.86 - np.array([49.79, 56.37, 54.79, 20.79, 75.10, 34.39, 31.25]))/152.86
    methane_duration_rmse = np.array([49.79, 56.37, 54.79, 20.79, 75.10, 34.39, 31.25])
    methane_duration = np.append(methane_duration, [4.69])

    methane_slope_trend = 100.0*(28.54 - np.array([14.77, 14.71, 14.69, 14.76, 14.88, 14.79]))/28.54
    methane_duration_trend = 100.0*(152.86 - np.array([78.31, 78.34, 78.29, 41.45, 70.10, 85.53]))/152.86
    methane_slope_trend_local_data = 100.0*(28.54 - np.array([14.56, 14.77, 15.14, 11.53, 12.02, 17.95]))/28.54
    methane_duration_trend_local_data = 100.0*(152.86 - np.array([34.46, 48.03, 37.92, 20.73, 38.34, 34.52]))/152.86
    
    nyse_slope = 100.0*(127.16 - np.array([90.76, 86.56, 89.31, 88.75, 86.62, 86.55, 86.89]))/127.16
    nyse_slope_rmse = np.array([90.76, 86.56, 89.31, 88.75, 86.62, 86.55, 86.89])
    nyse_slope = np.append(nyse_slope, [23.31])
    nyse_duration = 100.0*(0.33 - np.array([33.08, 0.41, 12.21, 0.29, 0.42, 0.42, 1.23]))/0.33
    nyse_duration_rmse = np.array([33.08, 0.41, 12.21, 0.29, 0.42, 0.42, 1.23])
    nyse_duration = np.append(nyse_duration, [25.09])
    nyse_slope_trend = 100.0*(127.16 - np.array([86.52, 86.84, 86.59, 86.52, 86.42, 103.65]))/127.16
    nyse_duration_trend = 100.0*(0.33 - np.array([0.67, 0.46, 0.36, 0.44, 0.43, 0.25]))/0.33
    nyse_slope_trend_local_data = 100.0*(127.16 - np.array([90.45, 86.50, 90.44, 86.53, 86.42, 86.54]))/127.16
    nyse_duration_trend_local_data = 100.0*(0.33 - np.array([25.34, 0.47, 14.05, 0.44, 0.41, 0.45]))/0.33

    jse_slope = 100.0*(31.01 - np.array([19.87, 19.83, 19.90, 20.21, 20.08, 20.01, 19.65]))/31.01
    jse_slope_rmse = np.array([19.87, 19.83, 19.90, 20.21, 20.08, 20.01, 19.65])
    jse_duration = 100.0*(17.97 - np.array([12.51, 12.68, 12.48, 12.67, 12.62, 12.85, 12.49]))/17.97
    jse_duration_rmse = np.array([12.51, 12.68, 12.48, 12.67, 12.62, 12.85, 12.49])
    jse_slope_trend = 100.0*(31.01 - np.array([20.67, 19.89, 21.17, 22.68, 19.93, 23.37]))/31.01
    jse_duration_trend = 100.0*(17.97 - np.array([12.75, 12.74, 12.63, 12.69, 12.64, 13.26]))/17.97
    jse_slope_trend_local_data = 100.0*(31.01 - np.array([21.13, 20.16, 21.41, 22.68, 19.93, 23.27]))/31.01
    jse_duration_trend_local_data = 100.0*(17.97 - np.array([12.59, 12.74, 12.71, 12.69, 12.65, 13.19]))/17.97

    algorithms = ["MLP", "LSTM", "CNN", "RF", "GBM", "SVR", "Our TreNet", "Lin et al."]
    runtime_index = ["MLP", "LSTM", "CNN", "RF", "GBM", "SVR", "Our TreNet"]
    trend_results = pd.DataFrame({"Voltage slope": voltage_slope,
                                  "Voltage duration": voltage_duration,
                                  "Methane slope": methane_slope,
                                  "Methane duration": methane_duration,
                                  "NYSE slope": nyse_slope,
                                  "NYSE duration": nyse_duration
                                  }, index=algorithms)

    """ CASH and BOHB """
    datasets = ["Methane", "Voltage", "NYSE", "JSE", "All"]
    # y_label = "Average slope RMSE and duration RMSE"
    # x_label = "Dataset/Average across datasets"
    # ''' CASH '''
    # cash_slope = np.array([14.93, 9.43, 86.61, 20.00])
    # cash_slope = np.append(cash_slope, [np.mean(cash_slope)])
    #
    # cash_duration = np.array([46.11, 62.88, 0.55, 12.46])
    # cash_duration = np.append(cash_duration, [np.mean(cash_duration)])
    #
    # best_bohb_slope_duration = np.array([14.01+40.09, 9.08+62.35, 86.62+0.72, 19.96+12.46])/2.0
    # best_bohb_slope_duration = np.append(best_bohb_slope_duration, [np.mean(best_bohb_slope_duration)])
    #
    # best_manual_slope_duration = np.array([14.57+49.79, 9.24+62.40, 86.56+0.41, 19.87+12.51])/2.0
    # best_manual_slope_duration = np.append(best_manual_slope_duration, [np.mean(best_manual_slope_duration)])
    #
    # cash_best_bohb_manual = pd.DataFrame({"best manual tuning": best_manual_slope_duration,
    #                                       "best BOHB": best_bohb_slope_duration,
    #                                       "CASH": (cash_slope + cash_duration)/2.0},
    #                                      index=datasets)
    # axes = cash_best_bohb_manual.plot.bar()
    #
    # axes.set_xlabel(x_label)
    # axes.set_ylabel(y_label)
    # END

    # ''' Comparison of BOHB and Manual '''

    manual_mlp_slope = np.array([14.23, 9.04, 90.76, 19.87])
    manual_mlp_slope = np.append(manual_mlp_slope, [np.mean(manual_mlp_slope)])
    bohb_mlp_slope = np.array([14.01, 8.94, 86.78, 20.11])
    bohb_mlp_slope = np.append(bohb_mlp_slope, [np.mean(bohb_mlp_slope)])
    manual_mlp_duration = np.array([49.79, 62.82, 33.08, 12.51])
    manual_mlp_duration = np.append(manual_mlp_duration, [np.mean(manual_mlp_duration)])
    bohb_mlp_duration = np.array([40.09, 62.75, 1.14, 12.43])
    bohb_mlp_duration = np.append(bohb_mlp_duration, [np.mean(bohb_mlp_duration)])

    bohb_mlp_manual = pd.DataFrame({"manual tuning": (manual_mlp_slope + manual_mlp_duration)/2.0,
                                        "BOHB": (bohb_mlp_slope + bohb_mlp_duration)/2.0},
                                       index=datasets)
    print("MLP")
    print(bohb_mlp_manual)
    # axes = bohb_mlp_manual.plot.bar()

    manual_lstm_slope = np.array([14.21, 10.30, 86.56, 19.83])
    manual_lstm_slope = np.append(manual_lstm_slope, [np.mean(manual_lstm_slope)])
    bohb_lstm_slope = np.array([14.20, 10.30, 86.62, 19.99])
    bohb_lstm_slope = np.append(bohb_lstm_slope, [np.mean(bohb_lstm_slope)])
    manual_lstm_duration = np.array([56.37, 62.87, 0.41, 12.68])
    manual_lstm_duration = np.append(manual_lstm_duration, [np.mean(manual_lstm_duration)])
    bohb_lstm_duration = np.array([54.10, 62.92, 0.72, 12.63])
    bohb_lstm_duration = np.append(bohb_lstm_duration, [np.mean(bohb_lstm_duration)])

    bohb_lstm_manual = pd.DataFrame({"manual tuning": (manual_lstm_slope + manual_lstm_duration)/2.0,
                                        "BOHB": (bohb_lstm_slope + bohb_lstm_duration)/2.0},
                                       index=datasets)
    # axes = bohb_lstm_manual.plot.bar()
    print("LSTM")
    print(bohb_lstm_manual)

    manual_cnn_slope = np.array([15.07, 9.24, 89.31, 19.90])
    manual_cnn_slope = np.append(manual_cnn_slope, [np.mean(manual_cnn_slope)])
    bohb_cnn_slope = np.array([15.50, 9.08, 86.60, 19.96])
    bohb_cnn_slope = np.append(bohb_cnn_slope, [np.mean(bohb_cnn_slope)])
    manual_cnn_duration = np.array([54.79, 62.40, 12.21, 12.48])
    manual_cnn_duration = np.append(manual_cnn_duration, [np.mean(manual_cnn_duration)])
    bohb_cnn_duration = np.array([47.80, 62.35, 1.05, 12.46])
    bohb_cnn_duration = np.append(bohb_cnn_duration, [np.mean(bohb_cnn_duration)])

    bohb_cnn_manual = pd.DataFrame({"manual tuning": (manual_cnn_slope + manual_cnn_duration) / 2.0,
                                    "BOHB": (bohb_cnn_slope + bohb_cnn_duration) / 2.0},
                                   index=datasets)
    # axes = bohb_cnn_manual.plot.bar()
    print("CNN")
    print(bohb_cnn_manual)

    manual_rf_slope = np.array([10.09, 9.53, 88.75, 20.21])
    manual_rf_slope = np.append(manual_rf_slope, [np.mean(manual_rf_slope)])
    bohb_rf_slope = np.array([10.07, 9.53, 87.84, 20.25])
    bohb_rf_slope = np.append(bohb_rf_slope, [np.mean(bohb_rf_slope)])
    manual_rf_duration = np.array([20.79, 63.11, 0.29, 12.67])
    manual_rf_duration = np.append(manual_rf_duration, [np.mean(manual_rf_duration)])
    bohb_rf_duration = np.array([29.72, 62.99, 0.27, 12.67])
    bohb_rf_duration = np.append(bohb_rf_duration, [np.mean(bohb_rf_duration)])

    bohb_rf_manual = pd.DataFrame({"manual tuning": (manual_rf_slope + manual_rf_duration)/2.0,
                                        "BOHB": (bohb_rf_slope + bohb_rf_duration)/2.0},
                                       index=datasets)
    # axes = bohb_rf_manual.plot.bar()
    print("RF")
    print(bohb_rf_manual)

    manual_gbm_slope = np.array([13.05, 10.0, 86.62, 20.08])
    manual_gbm_slope = np.append(manual_gbm_slope, [np.mean(manual_gbm_slope)])
    bohb_gbm_slope = np.array([13.57, 9.88, 86.61, 20.09])
    bohb_gbm_slope = np.append(bohb_gbm_slope, [np.mean(bohb_gbm_slope)])
    manual_gbm_duration = np.array([75.10, 62.67, 0.42, 12.62])
    manual_gbm_duration = np.append(manual_gbm_duration, [np.mean(manual_gbm_duration)])
    bohb_gbm_duration = np.array([76.67, 62.66, 0.41, 12.61])
    bohb_gbm_duration = np.append(bohb_gbm_duration, [np.mean(bohb_gbm_duration)])

    bohb_gbm_manual = pd.DataFrame({"manual tuning": (manual_gbm_slope + manual_gbm_duration)/2.0,
                                        "BOHB": (bohb_gbm_slope + bohb_gbm_duration)/2.0},
                                       index=datasets)
    axes = bohb_gbm_manual.plot.bar()
    print("GBM")
    print(bohb_gbm_manual)

    manual_svr_slope = np.array([14.98, 9.32, 86.55, 20.01])
    manual_svr_slope = np.append(manual_svr_slope, [np.mean(manual_svr_slope)])
    bohb_svr_slope = np.array([12.73, 9.32, 86.54, 20.09])
    bohb_svr_slope = np.append(bohb_svr_slope, [np.mean(bohb_svr_slope)])
    manual_svr_duration = np.array([34.39, 62.58, 0.42, 12.85])
    manual_svr_duration = np.append(manual_svr_duration, [np.mean(manual_svr_duration)])
    bohb_svr_duration = np.array([56.37, 62.58, 0.45, 12.68])
    bohb_svr_duration = np.append(bohb_svr_duration, [np.mean(bohb_svr_duration)])

    bohb_svr_manual = pd.DataFrame({"manual tuning": (manual_svr_slope + manual_svr_duration)/2.0,
                                        "BOHB": (bohb_svr_slope + bohb_svr_duration)/2.0},
                                       index=datasets)
    axes = bohb_svr_manual.plot.bar()
    print("SVR")
    print(bohb_svr_manual)
    #
    # axes.set_xlabel(x_label)
    # axes.set_ylabel(y_label)
    # END

    runtimes = pd.DataFrame({"Voltage": np.array([61.24, 17.36, 825.00, 1.44, 1.35, 15.68, 10]),
                              "Methane": np.array([3096.43, 1113.98, 1511, 13.72, 481.39, 3636.44, 10]),
                              "NYSE": np.array([7.45, 0.73, 114.60, 1.72, 1.65, 7.41, 10]),
                              "JSE": np.array([4.57, 2.43, 3.64, 19.63, 3.28, 20.15, 10])
                             }, index=runtime_index)
    #
    voltage_trend_results = pd.DataFrame({"Voltage slope": voltage_slope,
                                          "Voltage duration": voltage_duration}, index=algorithms)
    # methane_trend_results = pd.DataFrame({"Methane slope": methane_slope,
    #                                       "Methane duration": methane_duration}, index=algorithms)
    # nyse_trend_results = pd.DataFrame({"NYSE slope": nyse_slope,
    #                                    "NYSE duration": nyse_duration}, index=algorithms)
    # algorithms = ["MLP", "LSTM", "CNN", "RF", "GBM", "SVR", "Our TreNet"]
    # jse_trend_results = pd.DataFrame({"JSE slope": jse_slope,
    #                                   "JSE duration": jse_duration}, index=algorithms)
    # print(trend_results)
    # gradient_results.plot.bar()
    # plt.rcParams.update({'font.size': 10})

    ''' Comparison of vanilla algorithms '''
    # algorithms = ["MLP", "LSTM", "CNN", "RF", "GBM", "SVR", "Our TreNet"]
    # voltage_trend_rmses = pd.DataFrame({"Voltage slope": voltage_slope_rmse,
    #                                       "Voltage duration": voltage_duration_rmse}, index=algorithms)
    # methane_trend_rmses  = pd.DataFrame({"Methane slope": methane_slope_rmse,
    #                                       "Methane duration": methane_duration_rmse}, index=algorithms)
    # nyse_trend_rmses  = pd.DataFrame({"NYSE slope": nyse_slope_rmse,
    #                                    "NYSE duration": nyse_duration_rmse}, index=algorithms)
    # jse_trend_rmses  = pd.DataFrame({"JSE slope": jse_slope_rmse,
    #                                   "JSE duration": jse_duration_rmse}, index=algorithms)
    # val = jse_trend_rmses.drop(index=['Our TreNet'], inplace=False)
    # val = val.mean(axis=1)
    # # val = pd.DataFrame({"Point data": val.values,
    # #                     "Trend lines": (nyse_duration_trend + nyse_slope_trend)/2,
    # #                     "Trend lines + point data": (nyse_duration_trend_local_data + nyse_slope_trend_local_data)/2},
    # #                    index=val.index)
    # # val = val.sort_values(by="Point data", ascending=False)
    # val = val.sort_values(ascending=True)
    # print(val.shape)
    # axes = val.plot.bar()
    # axes = val.plot.bar(color='#FF7F50')
    # axes.set_ylabel("Average (slope and duration) RMSE")
    # axes.set_xlabel("Algorithms ranked per performance")

    ''' Comparison with TreNet and Lin et al. '''
    # x_label = 'Percentage improvement over naive last value prediction (%)'
    # y_label = 'Algorithms'
    # axes = voltage_trend_results.plot.barh()
    # axes.set_xlabel(x_label)
    # axes.set_ylabel(y_label)
    # axes = methane_trend_results.plot.barh()
    # axes.set_xlabel(x_label)
    # axes.set_ylabel(y_label)
    # axes = nyse_trend_results.plot.barh(subplots=False)
    # axes.set_xlabel(x_label)
    # # axes.set_ylabel(y_label)
    # axes = jse_trend_results.plot.barh(subplots=False)
    # axes.set_xlabel(x_label)
    # axes.set_ylabel(y_label)
    # gradient_results.plot.bar(subplots=True)
    # trend_results.plot.bar()

    ''' Average performance '''
    algorithms = ["MLP", "LSTM", "CNN", "RF", "GBM", "SVR", "Our TreNet"]
    trend_rmses = pd.DataFrame({"Voltage slope": voltage_slope_rmse,
                                  "Voltage duration": voltage_duration_rmse,
                                  "Methane slope": methane_slope_rmse,
                                  "Methane duration": methane_duration_rmse,
                                  "NYSE slope": nyse_slope_rmse,
                                  "NYSE duration": nyse_duration_rmse,
                                  "JSE slope": jse_slope_rmse,
                                  "JSE duration": jse_duration_rmse
                                }, index=algorithms)
    # print(trend_rmses.index)
    # trend_rmses.drop(index=['Lin et al.'], inplace=True)
    # trend_rmses['JSE slope'] = jse_slope_rmse
    # trend_rmses['JSE duration'] = jse_duration
    print('merged jse data')
    print(trend_rmses)
    avg = trend_rmses.mean(axis=1)
    # avg.drop(index=['Lin et al.'], inplace=True)
    std = trend_rmses.std(axis=1)
    # std.drop(index='Lin et al.', inplace=True)
    avg = pd.DataFrame({'Mean RMSE': avg, 'mean std': std,
                        'training time': runtimes.mean(axis=1), 'runtime std': runtimes.std(axis=1)},
                       index=avg.index).sort_values('Mean RMSE', ascending=True)
    print(avg)
    ax = avg[['Mean RMSE']].plot.bar(color='#FF7F50', yerr=avg['mean std'])
    # avg = avg['mean']
    ax.set_xlabel("Algorithms ranked per average performance")
    ax.set_ylabel('Average (slope and duration) RMSE')

    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width()*0.55, p.get_height() + 2), fontsize=10)

    # Time training time
    # avg.drop(index='Our TreNet', inplace=True)
    # avg = avg.sort_values('training time', ascending=True)
    # ax = avg[['training time']].plot.bar(color='#FF7F50', yerr=avg['runtime std'])
    # avg = avg['training time']
    # ax.set_xlabel("Algorithms ranked per average training time")
    # ax.set_ylabel('Training time (second)')
    # for idx, label in enumerate(list(avg.index)):
    #     value = np.round(avg.loc[label], decimals=2)
    #     ax.annotate(value, (idx, value), xytext=(0, 15), textcoords='offset points')


    # for i in ax.patches:
    #     # get_width pulls left or right; get_y pushes up or down
    #     ax.text(i. + .3, i.get_y() + .38,
    #             str(round(i.get_width(), 2)) + '%', fontsize=15,
    #             color='dimgrey')
    plt.show()

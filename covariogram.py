### Import Packages ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.spatial import distance_matrix
import seaborn as sns

#Define function instructions, to guide user to click bins
def instructions(s):
    print(s)
    plt.title(s, fontsize=20)
    plt.draw()

#Define Covariogram Function
def covariogram (v, x, y,  c0, c1, a, n_bins, site_name, transect, bin_method = "equal_n", model = "exponential"):
    #define array iter for iterating through dataset
    iter = range(0, len(x))
    # create a matrix which contain x and y positions from original data
    xy = np.array([x, y]).T

    # Compute Euclidian distances between all pairs and store in matrix
    distances = sp.spatial.distance_matrix(xy, xy, p=2)

    # Create matrix containing raw squared differences for each pair
    pair_diffs_matrix = np.zeros((len(x), len(x)))
    for i in iter:
        for j in iter:
            pair_diffs_matrix[i, j] = (v[j] - v[i])**2

    #Create arrays with only the upper right triangle of the pair diffs and distances matrices
    pair_diffs = pair_diffs_matrix[np.triu_indices(len(pair_diffs_matrix), 1)]
    distance_m = distances[np.triu_indices(len(pair_diffs_matrix), 1)]
    n = len(pair_diffs) - len(x)

    #Turn results into a dataframe
    diff_distance_list = [pair_diffs, distance_m]
    result_df = pd.DataFrame(diff_distance_list).transpose()
    result_df.columns = ["pair_diffs", "distance_m"]

    #Bin dataframe by number of bins that will contain equal number of values or by total distance divided by number of user input bins
    if isinstance(bin_method, int) or isinstance(bin_method, float):
        print("Bin method must be string 'equal_n', 'even_width', 'sqrt', or 'click_bins'")

    elif bin_method == "equal_n":
        bins = np.fromiter((np.nanpercentile(distance_m, (i / n_bins) * 100) for i in range(1, round(n_bins + 1))), dtype=float)

    elif bin_method == "sqrt":
        bins = np.fromiter((np.nanpercentile(distance_m, (i / math.sqrt(n)) * 100) for i in range(1, round(math.sqrt(n) + 1))), dtype=float)

    elif bin_method == "even_width":
        max_distance = np.nanmax(distance_m)
        bins = np.linspace(0, max_distance, n_bins + 1)[1:]

    elif bin_method == "click_bins":
        # plot raw pair differences against distance
        sns.scatterplot(x=distance_m, y=pair_diffs, data=result_df, marker='x', s=20, label="Raw Differences")
        plt.title("Raw Annual TP Deposition Squared Differences", fontsize=20)
        plt.xlabel("Euclidian Distance (m)", fontsize=16)
        plt.ylabel("(Vj - Vi)^2", fontsize=16)
        instructions("Click graph in ascending order " + str(n_bins) + " times to delineate bin width")
        clicks = plt.ginput(n_bins, timeout=0, show_clicks=True)
        instructions("Raw Annual TP Deposition Differences")
        bins = []
        for i in clicks:
            bins.append(i[0])
    else:
        bins = []

    bins = np.asarray(bins)
    result_df = result_df.sort_values(by=['distance_m'], ascending=True)
    result_df = result_df.reset_index(drop=True)

    cv_list = []
    lags = []
    n_per_bin = []
    std_cv_list = []

    for i in range(n_bins):
        if i == 0:
            bin_i = result_df.loc[(result_df['distance_m'] <= bins[i])]
            cv_list.append(np.nanmean(bin_i["pair_diffs"]) / 2)
            lags.append(np.nanmean(bin_i["distance_m"]))
            std_cv_list.append(np.nanstd(bin_i["pair_diffs"]))
            n_per_bin.append(len(bin_i))
        else:
            bin_i = result_df.loc[(result_df['distance_m'] <= bins[i]) & (result_df["distance_m"] > bins[i-1])]
            cv_list.append(np.nanmean(bin_i["pair_diffs"]) / 2)
            lags.append(np.nanmean(bin_i["distance_m"]))
            std_cv_list.append(np.nanstd(bin_i["pair_diffs"]))
            n_per_bin.append(len(bin_i))

    # define lists for storing model values for each bin, and upper and lower 95% CI
    upper_95ci = []
    lower_95ci = []
    exponential_model = []
    gaussian_model = []
    spherical_model = []


    for i in range(n_bins):
        if model == "exponential":
            exponential = c0 + c1 * (1 - math.e ** (-3 * abs(lags[i]) / a))
            exponential_model.append(exponential)
        elif model == "gaussian":
            gaussian = c0 + c1 * (1 - math.e ** (-(math.sqrt(3) * abs(lags[i]) / a) ** 2))
            gaussian_model.append(gaussian)
        elif model == "spherical":
            if lags[i] <= a:
                spherical = c0 + c1 * ((3 / 2) * (lags[i] / a) - (1 / 2) * ((lags[i] / a) ** 3))
                spherical_model.append(spherical)
            else:
                spherical = c0 + c1
                spherical_model.append(spherical)
        upper_ci = cv_list[i] + 1.96 * (std_cv_list[i] / math.sqrt(n_per_bin[i]))
        lower_ci = cv_list[i] - 1.96 * (std_cv_list[i] / math.sqrt(n_per_bin[i]))

        upper_95ci.append(upper_ci)
        lower_95ci.append(lower_ci)


    #Turn SV, mean, and max distances into a dataframe
    bin_values_list = [cv_list, lags, bins, n_per_bin, exponential_model, upper_95ci, lower_95ci, gaussian_model, spherical_model]
    bin_values_df = pd.DataFrame(bin_values_list).transpose()
    bin_values_df.columns = ['Mean CV', "Mean Distance", "Max Distance", "n", "Exponential Model", "Upper 95% CI", "Lower 95% CI", "Gaussian Model", "Spherical Model"]

    #Plot raw semivariogram and normal semivariogram

    # plt.vlines(bins, 0, 1000,  colors="grey", linestyles="dashed", alpha=0.25, label="Bins")
    sns.scatterplot(x=distance_m, y=pair_diffs, data=result_df, marker='x', s=20, label="Raw Differences")
    palette = sns.color_palette("Reds", as_cmap=True)
    sns.scatterplot(x=lags, y=cv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Covariance")
    for i, txt in enumerate(n_per_bin):
        plt.annotate(txt, (lags[i], cv_list[i]), (lags[i] + .1, cv_list[i] + .1))
    plt.title("Raw TP Deposition Squared Differences and Covariance for " + str(site_name) + ": " + str(transect), fontsize=20)
    plt.xlabel("Euclidian Distance (m)", fontsize=16)
    plt.ylabel("(Vj - Vi)^2", fontsize=16)
    plt.legend(loc="upper right")
    plt.show()

    #Plot Semivariogram with Exponential model and 95% CIs
    #R2 calculation
    if model == "exponential":
        RSS_ex = 0
        TSS_ex = 0
        for i in range(len(cv_list)):
            RSS_ex += (cv_list[i] - exponential_model[i]) ** 2
            TSS_ex += (cv_list[i] - np.mean(cv_list)) ** 2
        R2_ex = 1 - (RSS_ex/TSS_ex)
        sns.scatterplot(x=lags, y=cv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        for i, txt in enumerate(n_per_bin):
            plt.annotate(txt, (lags[i], cv_list[i]), (lags[i] + .1, cv_list[i] + .1))
        sns.lineplot(data=bin_values_df, x=lags, y=exponential_model, label="Exponential Model")
        plt.plot(lags, upper_95ci, color="#a90505", label="95% Confidence Interval")
        plt.plot(lags, lower_95ci, color="grey")
        plt.title("Semivariogram of TP Deposition w/ Exponential Model for " + str(site_name) + ": " + str(transect), fontsize=20)
        plt.xlabel("Euclidian Distance (m)", fontsize=16)
        plt.ylabel("\u03B3(h)", fontsize=16)
        plt.legend(loc="upper right")
        #plt.text(25, 15, "R^2 = " + str(round(R2_ex, 3)))
        plt.show()

    # Plot Semivariogram with Gaussian model and 95% CIs
    #R2 Calculation
    if model == "gaussian":
        RSS_ga = 0
        TSS_ga = 0
        for i in range(len(cv_list)):
            RSS_ga += (cv_list[i] - gaussian_model[i]) ** 2
            TSS_ga += (cv_list[i] - np.mean(cv_list)) ** 2
        R2_ga = 1 - (RSS_ga/TSS_ga)
        sns.scatterplot(x=lags, y=cv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        for i, txt in enumerate(n_per_bin):
            plt.annotate(txt, (lags[i], cv_list[i]), (lags[i] + .1, cv_list[i] + .1))
        sns.lineplot(data=bin_values_df, x=lags, y=exponential_model, label="Gaussian Model")
        plt.plot(lags, upper_95ci, color="#a90505", label="95% Confidence Interval")
        plt.plot(lags, lower_95ci, color="#a90505")
        plt.title("Semivariogram of Annual TP Deposition w/ Gaussian Model for " + str(site_name) + ": " + str(transect), fontsize=20)
        plt.xlabel("Euclidian Distance (m)", fontsize=16)
        plt.ylabel("\u03B3(h)", fontsize=16)
        plt.legend(loc="upper right")
        #plt.text(25, 15, "R^2 = " + str(round(R2_ga, 3)))
        plt.show()

    # Plot Semivariogram with Spherical model and 95% CIs
    #R2 Calculation
    if model == "spherical":
        RSS_sp = 0
        TSS_sp = 0
        for i in range(len(cv_list)):
            RSS_sp += (cv_list[i] - spherical_model[i]) ** 2
            TSS_sp += (cv_list[i] - np.mean(cv_list)) ** 2
        R2_sp = 1-(RSS_sp/TSS_sp)
        sns.scatterplot(x=lags, y=cv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        for i, txt in enumerate(n_per_bin):
            plt.annotate(txt, (lags[i], cv_list[i]), (lags[i] + .1, cv_list[i] + .1))
        sns.lineplot(data=bin_values_df, x=lags, y=spherical_model, label="Spherical Model")
        plt.plot(lags, upper_95ci, color="#a90505", label="95% Confidence Interval")
        plt.plot(lags, lower_95ci, color="#a90505")
        plt.title("Semivariogram of Annual TP Deposition w/ Spherical Model for " + str(site_name) + ": " + str(transect), fontsize=20)
        plt.xlabel("Euclidian Distance (m)", fontsize=16)
        plt.ylabel("\u03B3(h)", fontsize=16)
        plt.legend(loc="upper right")
        #plt.text(25, 15, "R^2 = " + str(round(R2_sp, 3)))
        plt.show()


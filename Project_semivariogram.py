### CE 369 - Project Semivariogram code ###
### Import Packages ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy as sp
import math

import scipy.stats
import skgstat as skg
from scipy.spatial import distance_matrix
import seaborn as sns

# set working directory to downloaded folder
os.chdir(r'C:\Users\ghmye\OneDrive\Documents\UVM\CE369\Project_Work\python_data')

#Import data from excel into python and check first and last five rows of dataset
data = pd.read_excel(r'mckenzie_110119.xls')
data = data.dropna()

### DATA MANAGEMENT ###
#Rename columns for ease of coding
data = data.rename(columns={"TP Deposition (g P / m2)": "tp_dep", "Sediment Deposition (kg/m2)": "sed_dep", "Distance from Channel (m)": "dist", "Stdev Sediment Deposition (g cm-2)": "stdev"})

#Create subdataframes for each site and event w/ over 8 plots
transect1 = data.loc[(data["Plot"] == "Plot1") | (data["Plot"] == "Plot2") | (data["Plot"] == "Plot3")]
transect1 = transect1.reset_index(drop=True)

#define nugget as the average standard deviation
nugget = np.nanmean(transect1["stdev"])

#Define variables of interest
v1 = transect1["tp_dep"]

# plt.hist(v1)
# plt.title("Distribution of TP deposition")
# plt.ylabel("Count")
# plt.xlabel("TP Deposition (g/m2)")
# plt.show()

#Define Location vectors
x = transect1["dist"]
elev = transect1["elev"]
#y = transect1["y"]
logx = np.log10(x)
logelev = np.log10(elev)
xy1 = np.array([x, elev]).T
xy1_log = np.array([logx, logelev]).T


# variogram = skg.Variogram(xy1, v1, model='gaussian')
# variogram.plot()
# plt.show()

#Define function instructions, to guide user to click bins
def instructions(s) :
    print(s)
    plt.title(s, fontsize=20)
    plt.draw()

#Define Semivariogram Function
def semivariogram (v, x, y, c0, c1, a, n_bins, bin_method = "equal_n", model = "exponential"):
    #Take log of primary variable to account for non-normal distribution
    log_v = np.log10(v)
    p_val = scipy.stats.shapiro(v)[1]

    #define array iter for iterating through dataset
    iter = range(0, len(x))

    # create a matrix which contain x and y positions from original data
    xy = np.array([x, y]).T

    # Compute Euclidian distances between all points
    distances = sp.spatial.distance_matrix(xy, xy, p=2)

    # Create matrix containing raw squared differences for each pair
    pair_diffs_matrix = np.zeros((len(x), len(x)))
    log_pair_diffs_matrix = np.zeros((len(x), len(x)))
    for i in iter:
        for j in iter:
            pair_diffs_matrix[i, j] = (v[j] - v[i])**2
            log_pair_diffs_matrix[i, j] = (log_v[j] - log_v[i]) ** 2

    #Create arrays with only the upper triangle of the pair diffs and distances matrices
    pair_diffs = pair_diffs_matrix[np.triu_indices(len(pair_diffs_matrix), 1)]
    log_pair_diffs = log_pair_diffs_matrix[np.triu_indices(len(log_pair_diffs_matrix), 1)]
    distance_m = distances[np.triu_indices(len(log_pair_diffs_matrix), 1)]
    n = len(pair_diffs)

    #Turn results into a dataframe
    diff_distance_list = [pair_diffs, log_pair_diffs, distance_m]
    result_df = pd.DataFrame(diff_distance_list).transpose()
    result_df.columns = ["pair_diffs", "log_pair_diffs", "distance_m"]

    #Bin dataframe by number of bins that will contain equal number of values or by total distance divided by number of user input bins
    if isinstance(bin_method, int) or isinstance(bin_method, float):
        print("Bin method must be string 'equal_n', 'even_width', 'sqrt', or 'click_bins'")

    elif bin_method == "equal_n":
        bins = np.fromiter((np.nanpercentile(distance_m, (i / n_bins) * 100) for i in range(1, round(n_bins + 1))), dtype=float)

    elif bin_method == "sqrt":
        bins = np.fromiter((np.nanpercentile(distance_m, (i / round(math.sqrt(n))) * 100) for i in range(1, round(math.sqrt(n) + 1))), dtype=float)
        n_bins = len(bins)

    elif bin_method == "even_width":
        max_distance = np.nanmax(distance_m)
        bins = np.linspace(0, max_distance, n_bins + 1)[1:]
        n_bins = len(bins)

    elif bin_method == "doane":
        bins = np.histogram_bin_edges(distance_m, bins="doane")
        n_bins = len(bins)

    elif bin_method == "click_bins":
        # plot raw pair differences against distance
        sns.scatterplot(x=distance_m, y=pair_diffs, data=result_df, marker='s', s=20, label="Raw Differences")
        plt.title("Raw Annual TP Deposition Squared Differences", fontsize=20)
        plt.xlabel("Euclidian Distance (m)", fontsize=16)
        plt.ylabel("(Vj - Vi)^2", fontsize=16)
        instructions("Click graph in ascending order " + str(n_bins) + " times to delineate bin width")
        clicks = plt.ginput(n_bins, timeout=0, show_clicks=True)
        instructions("Raw Annual TP Deposition Differences")
        bins = []
        for i in clicks:
            bins.append(i[0])

    bins = np.asarray(bins)
    result_df = result_df.sort_values(by=['distance_m'], ascending=True)
    result_df = result_df.reset_index(drop=True)

    #Create empty lists for storing semivariance, lags, number of pairs per bin, and standard deviation
    sv_list = []
    lags = []
    n_per_bin = []
    std_sv_list = []

    for i in range(n_bins):
        if i == 0:
            bin_i = result_df.loc[(result_df['distance_m'] <= bins[i])]
            if p_val >= 0.05:
                sv_list.append(np.nanmean(bin_i["pair_diffs"]) / 2)
                std_sv_list.append(np.nanstd(bin_i["pair_diffs"]))
            elif p_val < 0.05:
                sv_list.append(np.nanmean(bin_i["log_pair_diffs"]) / 2)
                std_sv_list.append(np.nanstd(bin_i["log_pair_diffs"]))
            lags.append(np.nanmean(bin_i["distance_m"]))
            n_per_bin.append(len(bin_i))
        else:
            bin_i = result_df.loc[(result_df['distance_m'] <= bins[i]) & (result_df["distance_m"] > bins[i-1])]
            sv_list.append(np.nanmean(bin_i["pair_diffs"]) / 2)
            lags.append(np.nanmean(bin_i["distance_m"]))
            std_sv_list.append(np.nanstd(bin_i["pair_diffs"]))
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
        upper_ci = sv_list[i] + 1.96 * (std_sv_list[i] / math.sqrt(n_per_bin[i]))
        lower_ci = sv_list[i] - 1.96 * (std_sv_list[i] / math.sqrt(n_per_bin[i]))
        upper_95ci.append(upper_ci)
        lower_95ci.append(lower_ci)

    #Turn SV, mean, and max distances into a dataframe
    bin_values_list = [sv_list, lags, bins, n_per_bin, exponential_model, upper_95ci, lower_95ci, gaussian_model, spherical_model]
    bin_values_df = pd.DataFrame(bin_values_list).transpose()
    bin_values_df.columns = ['Mean SV', "Mean Distance", "Max Distance", "n", "Exponential Model", "Upper 95% CI", "Lower 95% CI", "Gaussian Model", "Spherical Model"]

    #Plot raw semivariogram and normal semivariogram

    #sns.set(rc={'figure.figsize': (11.7, 8.27)})
    # plt.vlines(bins, 0, 500,  colors="grey", linestyles="dashed", alpha=0.25, label="Bins")
    sns.scatterplot(x=distance_m, y=pair_diffs, data=result_df, marker='s', s=20, label="Raw Differences")
    palette = sns.color_palette("Reds", as_cmap=True)
    sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
    for i, txt in enumerate(n_per_bin):
        plt.annotate(txt, (lags[i], sv_list[i]), (lags[i] + .1, sv_list[i] + .1))
    if p_val >= 0.05:
        plt.title("Raw TP Deposition Squared Differences and Semivariogram", fontsize=20)
    else:
        plt.title("Raw TP Deposition Squared Differences and Log Transformed Semivariogram", fontsize=20)
    plt.xlabel("Euclidian Distance (m)", fontsize=16)
    plt.ylabel("(Vj - Vi)^2", fontsize=16)
    plt.legend(loc="upper right")
    plt.show()

    # Plot Semivariogram with Exponential model and 95% CIs
    # R2 calculation
    gs = dict(height_ratios=[1, 4])
    if model == "exponential":
        RSS_ex = 0
        TSS_ex = 0
        for i in range(len(sv_list)):
            RSS_ex += (sv_list[i] - exponential_model[i]) ** 2
            TSS_ex += (sv_list[i] - np.mean(sv_list)) ** 2
        R2_ex = 1 - (RSS_ex / TSS_ex)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if p_val >= 0.05:
            plt.title("Semivariogram of TP Deposition w/ Exponential Model", fontsize=20)
        else:
            plt.title("Semivariogram of Log TP Deposition w/ Exponential Model", fontsize=16)
        sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        # plt.vlines(bins, 0, 500, colors="grey", linestyles="dashed", alpha=0.25, label="Bins")
        sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        for i, txt in enumerate(n_per_bin):
            plt.annotate(txt, (lags[i], sv_list[i]), (lags[i] + .1, sv_list[i] + .1))
        sns.lineplot(data=bin_values_df, x=lags, y=exponential_model, palette=palette, label="Exponential Model")
        plt.plot(lags, upper_95ci, color="black", alpha=0.6, label="95% Confidence Interval")
        plt.plot(lags, lower_95ci, color="black", alpha=0.6)
        plt.xlabel("Euclidian Distance (m)", fontsize=12)
        plt.ylabel("\u03B3(h)", fontsize=12)
        plt.legend(loc="upper right")
        # plt.text(25, 15, "R^2 = " + str(round(R2_ex, 3)))
        plt.show()

    # Plot Semivariogram with Gaussian model and 95% CIs
    # R2 Calculation
    if model == "gaussian":
        RSS_ga = 0
        TSS_ga = 0
        for i in range(len(sv_list)):
            RSS_ga += (sv_list[i] - gaussian_model[i]) ** 2
            TSS_ga += (sv_list[i] - np.mean(sv_list)) ** 2
        R2_ga = 1 - (RSS_ga / TSS_ga)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        # plt.vlines(bins, 0, 500, colors="grey", linestyles="dashed", alpha=0.25, label="Bins")
        sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        for i, txt in enumerate(n_per_bin):
            plt.annotate(txt, (lags[i], sv_list[i]), (lags[i] + .1, sv_list[i] + .1))
        sns.lineplot(data=bin_values_df, x=lags, y=gaussian_model, palette=palette,  label="Gaussian Model")
        plt.plot(lags, upper_95ci, color="black", label="95% Confidence Interval", alpha=0.6)
        plt.plot(lags, lower_95ci, color="black", alpha=0.6)
        if p_val >= 0.05:
            plt.title("Semivariogram of TP Deposition w/ Gaussian Model", fontsize=16)
        else:
            plt.title("Semivariogram of Log TP Deposition w/ Gaussian Model", fontsize=16)
        plt.xlabel("Euclidian Distance (m)", fontsize=12)
        plt.ylabel("\u03B3(h)", fontsize=12)
        plt.legend(loc="upper right")
        # plt.text(25, 15, "R^2 = " + str(round(R2_ga, 3)))
        plt.show()

    # Plot Semivariogram with Spherical model and 95% CIs
    # R2 Calculation
    if model == "spherical":
        RSS_sp = 0
        TSS_sp = 0
        for i in range(len(sv_list)):
            RSS_sp += (sv_list[i] - spherical_model[i]) ** 2
            TSS_sp += (sv_list[i] - np.mean(sv_list)) ** 2
        R2_sp = 1 - (RSS_sp / TSS_sp)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        # plt.vlines(bins, 0, 10, colors="grey", linestyles="dashed", alpha=0.25, label="Bins")
        sns.scatterplot(x=lags, y=sv_list, data=bin_values_df, marker='o', palette=palette, s=40, label="Semivariance")
        for i, txt in enumerate(n_per_bin):
            plt.annotate(txt, (lags[i], sv_list[i]), (lags[i] + .1, sv_list[i] + .1))
        sns.lineplot(data=bin_values_df, x=lags, y=spherical_model, palette=palette, label="Spherical Model")
        plt.plot(lags, upper_95ci, color="black", label="95% Confidence Interval", alpha=0.6)
        plt.plot(lags, lower_95ci, color="black", alpha=0.6)
        if p_val >= 0.05:
            plt.title("Semivariogram of TP Deposition w/ Spherical Model", fontsize=16)
        else:
            plt.title("Semivariogram of Log TP Deposition w/ Spherical Model", fontsize=16)
        plt.xlabel("Euclidian Distance (m)", fontsize=12)
        plt.ylabel("\u03B3(h)", fontsize=12)
        plt.legend(loc="upper right")
        # plt.text(25, 15, "R^2 = " + str(round(R2_sp, 3)))
        plt.show()

    # Distribution of Bins over Distance
    # sns.histplot(data=bin_values_df, x=bins)
    # plt.title("Distribution of Bins over Euclidian Distance", fontsize=20)
    # plt.xlabel("Euclidian Distance (m)", fontsize=16)
    # plt.show()


semivariogram(v1, x, elev, nugget, 200, 19, 5, "equal_n", "gaussian")

def ordinary_kriging(v, x, y, x_est_bounds, y_est_bounds, c0, c1, a):
    #define n as length of data
    n = len(x)
    # create a matrix which contain x and y positions from original data
    xy1 = np.array([x, y]).T

    # Compute Euclidian distances between all known points
    distances = sp.spatial.distance_matrix(xy1, xy1, p=2)

    #Create an empty n + 1 x n + 1 matrix for storing semivariance and covariance values
    sv_ij = np.ones((n + 1, n + 1))
    cv_ij = np.ones((n + 1, n + 1))
    for i in range(0, n):
        for j in range(0, n):
            sv_ij[i, j] = c0 + c1 * (1 - math.e ** (-3 * abs(distances[i, j]) / a))
            cv_ij[i, j] = c0 + c1 * math.e**(-3 * abs(distances[i, j]) / a)

    #set the last element of the matrix equal to 0
    sv_ij[n, n] = 0
    cv_ij[n, n] = 0

    # Create Dataframe which contains all unknown points for estimation
    # Create a matrix of the x and y-coordinates at which to estimate concentration
    xx, yy = np.meshgrid(np.linspace(x_est_bounds[0], x_est_bounds[1], x_est_bounds[1]*2 + 1), np.linspace(y_est_bounds[0], y_est_bounds[1], y_est_bounds[1]*2 + 1))

    #Reshape xx and yy for scatterplots if making
    # x_estimate = np.reshape(xx, (xx.shape[0]*xx.shape[1]))
    # y_estimate = np.reshape(yy, (yy.shape[0]*yy.shape[1]))

    # Create d_io - 1x5 vector which contains distances from known points to desired estimation point
    v_io_matrix = np.zeros((xx.shape[0], xx.shape[1]))
    error_variance_matrix = np.zeros((xx.shape[0], yy.shape[1]))
    for i in range(0, len(xx)):
        for j in range(0, yy.shape[1]-1):
            d_io = np.ones(n)
            for z in range(0, n):
                d_io[z] = math.dist((x[z], y[z]), (xx[i, j], yy[i, j]))

    #Create sv_io - 1 x len(x)+1 vector which contains modeled semivariance values between known points and estimated point
            sv_io = np.ones(n+1)
            for z in range(0, n):
                sv_io[z] = c0 + c1 * (1 - math.e ** (-3 * abs(d_io[z]) / a))

            #Create wts vector by multiplying the inverse of the sv_ij matrix by sv_io vector
            wts = np.linalg.lstsq(sv_ij, sv_io, rcond=None)
            wts = wts[0]

            #wts_lagrange is 1xlen(x)+1 and contains lagrange parameter, wts is 1xlen(x) and does not
            wts_lagrange = wts.tolist()
            wts = wts.tolist()
            wts.pop()

            #Estimate concentration and error at unknown points by taking the sum of the products of the weights and known values
            v_io = 0
            error_io = 0

            for z in range(len(wts)):
                v_io += wts[z] * v[z]
                error_io += np.dot(sv_io, wts_lagrange)
            if v_io >= 0:
                v_io_matrix[i, j] = v_io
                error_variance_matrix[i, j] = error_io
            else:
                v_io_matrix[i, j] = 0
                error_variance_matrix[i, j] = error_io

    #Reshape matrices to arrays if you want to create a scatterplot
    # v_io_array = np.reshape(v_io_matrix, (xx.shape[0]*xx.shape[1]))
    # error_variance_array = np.reshape(error_variance_matrix, (xx.shape[0]*xx.shape[1]))

    #Create a dataframe of all estimated concentrations and error, only needed for scatter plots
    # estimation_points = [x_estimate, y_estimate, v_io_array, error_variance_array]
    # estimation_points = pd.DataFrame(estimation_points).transpose()
    # estimation_points.columns = ["x", "y", "V estimated", "Modeled Error Variance"]

    # #Create ticks for scatter plot if making
    # xticks = range(x_est_bounds[0], x_est_bounds[1] + 1)
    # yticks = range(y_est_bounds[0], y_est_bounds[1] + 1)

    #Scatter Plots
    #Create Estimated Values Plot
    # plot1 = plt.figure(1)
    # conc_palette = sns.color_palette("RdPu", as_cmap=True)
    # error_palette = sns.color_palette("RdYlGn_r", as_cmap=True)
    # sns.scatterplot(data=estimation_points, x=x_estimate, y=y_estimate, palette=conc_palette, hue=v_io_array, marker='s', s=100, label="Estimated Points")
    # plt.scatter(x, y, color="blue", marker='x', s=100, label="Known Points")
    # plt.legend(loc="lower right")
    # plt.suptitle("Estimated Concentrations over 2-D Space", fontsize=12)
    # plt.title("Nugget = 0, Sill = 15, Range = 10", fontsize=8)
    # plt.ylabel("mm")
    # plt.yticks(yticks)
    # plt.xlabel("mm")
    # plt.xticks(xticks)
    # #plt.show()
    #
    # #Create Estimated Error Plot
    # plot2 = plt.figure(2)
    # sns.scatterplot(data=estimation_points, x=x_estimate, y=y_estimate, palette=error_palette, hue=error_variance_array, marker='s', s=100, label="Estimated Error")
    # plt.scatter(x, y, color="blue", marker='x', s=100, label="Known Points")
    # plt.legend(loc="lower right")
    # plt.suptitle("Estimated Error Variance over 2-D Space", fontsize=12)
    # plt.title("Nugget = 0, Sill = 15, Range = 10", fontsize=8)
    # plt.ylabel("mm")
    # plt.yticks(yticks)
    # plt.xlabel("mm")
    # plt.xticks(xticks)
    # #plt.show()

    #Contour Plots
    fig, ax = plt.subplots(figsize=(6, 4))
    contour = ax.contour(xx, yy, v_io_matrix, 10, extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]), colors="gray")
    ax.clabel(contour, inline=True, fontsize=8)
    im = ax.imshow(v_io_matrix, interpolation='none', origin="lower", cmap="RdPu", extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]))
    CB = fig.colorbar(im, shrink=0.8)
    CB.set_label("Estimated TP Deposition (g/m2)", rotation=270, labelpad=10)
    plt.suptitle("Estimated Concentrations over 2-D Space", fontsize=12)
    plt.title("Nugget = " + str(round(c0, 2)) + " Sill = " + str(c1) + " Range = " + str(a), fontsize=10)
    plt.xlabel("Distance from Channel (m)")
    plt.ylabel("Elevation above Channel (m)")
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    error_contour = ax1.contour(xx, yy, error_variance_matrix, 10, extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]), colors="gray")
    ax1.clabel(error_contour, inline=True, fontsize=8)
    error_im = ax1.imshow(error_variance_matrix, interpolation='none', origin="lower", cmap="RdYlGn_r", extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]))
    CB = fig1.colorbar(error_im, shrink=0.8)
    CB.set_label("Estimated Error Variance", rotation=270, labelpad=10)
    plt.suptitle("Estimated Error over 2-D Space", fontsize=12)
    plt.title("Nugget = " + str(round(c0, 2)) + " Sill = " + str(c1) + " Range = " + str(a), fontsize=10)
    plt.xlabel("Distance from Channel (m)")
    plt.ylabel("Elevation above Channel (m)")
    plt.show()

ordinary_kriging(v1, x, elev, [0, round(np.max(x)+10)], [0, round(np.max(elev) + 6)], nugget, 200, 19)
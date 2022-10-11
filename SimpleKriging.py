### Simple Kriging ###
### Import Packages ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy as sp
import math

import scipy.stats
from scipy.stats import exponpow
import skgstat as skg
from scipy.spatial import distance_matrix
from sklearn import linear_model
import seaborn as sns

#Define Simple Kriging Function
def simple_kriging(v, x, y, x_est_bounds, y_est_bounds, c0, c1, a, site_name, transect, model="exponential"):
    #define n as length of data
    n = len(x)

    #define mean and detrended data
    mean = np.mean(v)
    v_dt = np.zeros(len(v))
    for i in range(len(v)):
        v_dt[i] = v[i] - mean

    # create a matrix which contain x and y positions from original data
    xy1 = np.array([x, y]).T

    # Compute Euclidian distances between all known points
    distances = sp.spatial.distance_matrix(xy1, xy1, p=2)

    #Create an empty n x n matrix for storing semivariance and covariance values
    sv_ij = np.ones((n, n))
    cv_ij = np.ones((n, n))
    for i in range(0, n):
        for j in range(0, n):
            sv_ij[i, j] = c0 + c1 * (1 - math.e ** (-3 * abs(distances[i, j]) / a))
            cv_ij[i, j] = c0 + c1 * math.e**(-3 * abs(distances[i, j]) / a)

    # Create Dataframe which contains all unknown points for estimation
    # Create a matrix of the x and y-coordinates at which to estimate concentration
    xx, yy = np.meshgrid(np.linspace(x_est_bounds[0], x_est_bounds[1], x_est_bounds[1]*2 + 1), np.linspace(y_est_bounds[0], y_est_bounds[1], y_est_bounds[1]*2 + 1))

    #Reshape xx and yy for scatterplots if making
    # x_estimate = np.reshape(xx, (xx.shape[0]*xx.shape[1]))
    # y_estimate = np.reshape(yy, (yy.shape[0]*yy.shape[1]))

    # Create d_io - 1xn vector which contains distances from known points to desired estimation point
    v_io_matrix = np.zeros((xx.shape[0], xx.shape[1]))
    error_variance_matrix = np.zeros((xx.shape[0], yy.shape[1]))
    for i in range(0, len(xx)):
        for j in range(0, yy.shape[1]-1):
            d_io = np.ones(n)
            for z in range(0, n):
                d_io[z] = math.dist((x[z], y[z]), (xx[i, j], yy[i, j]))

    #Create sv_io - 1 x n vector which contains modeled semivariance values between known points and estimated point
            sv_io = np.ones(n)
            if model == "exponential":
                for z in range(0, n):
                    sv_io[z] = c0 + c1 * (1 - math.e ** (-3 * abs(d_io[z]) / a))
            elif model == "gaussian":
                for z in range(0, n):
                    sv_io[z] = c0 + c1 * (1 - math.e ** (-(math.sqrt(3) * abs(d_io[z]) / a) ** 2))
            elif model == "spherical":
                if d_io[z] <= a:
                    sv_io[z] = c0 + c1 * ((3 / 2) * (d_io[z] / a) - (1 / 2) * ((d_io[z] / a) ** 3))
                else:
                    sv_io[z] = c0 + c1

            #Create wts vector by multiplying the inverse of the sv_ij matrix by sv_io vector
            wts = np.linalg.lstsq(sv_ij, sv_io, rcond=None)
            wts = wts[0]

            #Estimate concentration and error at unknown points by taking the sum of the products of the weights and known values
            v_io = 0
            error_io = 0

            for z in range(len(wts)):
                v_io += wts[z] * v_dt[z]
            v_io += mean
            error_io += np.dot(sv_io, wts)

            v_io_matrix[i, j] = v_io
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
    contour = ax.contour(xx, yy, v_io_matrix, 14, extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]), colors="gray")
    ax.clabel(contour, inline=True, fontsize=6, colors='black')
    im = ax.imshow(v_io_matrix, interpolation='none', origin="lower", cmap="RdPu", extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]))
    plt.scatter(x, y, marker="x", color="blue", s=18)
    CB = fig.colorbar(im, shrink=0.8)
    CB.set_label("Estimated Log TP Deposition (g/m2)", rotation=270, labelpad=12, fontsize=14)
    plt.suptitle("Estimated Deposition over 2-D Space for " + str(site_name) + ": " + str(transect), fontsize=18)
    plt.tight_layout(pad=0.5)
    plt.title("Nugget = " + str(round(c0, 2)) + " Sill = " + str(c1) + " Range = " + str(a), fontsize=14)
    plt.xlabel("Distance from Channel (m)", fontsize=14)
    plt.ylabel("Elevation above Channel (m)", fontsize=14)
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    error_contour = ax1.contour(xx, yy, error_variance_matrix, 10, extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]), colors="gray")
    ax1.clabel(error_contour, inline=True, fontsize=6, colors='black')
    error_im = ax1.imshow(error_variance_matrix, interpolation='none', origin="lower", cmap="RdYlGn_r", extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]))
    plt.scatter(x, y, marker="x", color="blue", s=18)
    CB = fig1.colorbar(error_im, shrink=0.8)
    CB.set_label("Estimated Error Variance", rotation=270, labelpad=12, fontsize=14)
    plt.suptitle("Estimated Error over 2-D Space for " + str(site_name) + ": " + str(transect), fontsize=18)
    plt.tight_layout(pad=0.5)
    plt.title("Nugget = " + str(round(c0, 2)) + " Sill = " + str(c1) + " Range = " + str(a), fontsize=14)
    plt.xlabel("Distance from Channel (m)", fontsize=14)
    plt.ylabel("Elevation above Channel (m)", fontsize=14)
    plt.show()

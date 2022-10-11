#Import Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.spatial import distance_matrix

### Define Ordinary Kriging Function ###

#v is the variable of interest that we are trying to estimate
#x is the x coordinates associated with the known points
#y is the y coordinates associated with the known points
# x_est_bounds and y_est bounds are both lists containing the upper and lower bounds of the area we would like to estimate over
#c0 is the nugget
#c1 is the sill
#a is the range
def ordinary_kriging (v, x, y, x_est_bounds, y_est_bounds, c0, c1, a, model="exponential"):
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

    #Reshape xx and yy for
    x_estimate = np.reshape(xx, (xx.shape[0]*xx.shape[1]))
    y_estimate = np.reshape(yy, (yy.shape[0]*yy.shape[1]))

    # Create d_io - 1x5 vector which contains distances from known points to desired estimation point
    v_io_matrix = np.zeros((len(xx), len(yy)))
    error_variance_matrix = np.zeros((len(xx), len(yy)))
    for i in range(0, len(xx)):
        for j in range(0, yy.shape[1]-1):
            d_io = np.ones(len(x))
            for z in range(0, n):
                d_io[z] = math.dist((x[z], y[z]), (xx[i, j], yy[i, j]))

    #Create sv_io - 1 x 6 vector which contains modeled semivariance values between known points and estimated point
            sv_io = np.ones(len(x) + 1)
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
            wts = np.linalg.solve(sv_ij, sv_io)

            #wts_lagrange is 1x6 and contains lagrange parameter, wts is 1x5 and does not
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
    v_io_array = np.reshape(v_io_matrix, len(v_io_matrix)**2)
    error_variance_array = np.reshape(error_variance_matrix, len(error_variance_matrix)**2)

    #Create a dataframe of all estimated concentrations and error
    estimation_points = [x_estimate, y_estimate, v_io_array, error_variance_array]
    estimation_points = pd.DataFrame(estimation_points).transpose()
    estimation_points.columns = ["x", "y", "V estimated", "Modeled Error Variance"]

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
    fig, ax = plt.subplots()
    contour = ax.contour(xx, yy, v_io_matrix, 20, extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]), colors="gray")
    ax.clabel(contour, inline=True, fontsize=8)
    im = ax.imshow(v_io_matrix, interpolation='none', origin="lower", cmap="RdPu", extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]))
    plt.scatter(x, y, marker="x", color="blue", s=18)
    CB = fig.colorbar(im, shrink=0.8)
    CB.set_label("Estimated Concentration (mg/L)", rotation=270, labelpad=10)
    plt.suptitle("Estimated Concentrations over 2-D Space", fontsize=12)
    plt.title("Nugget = " + str(c0) + " Sill = " + str(c1) + " Range = " + str(a), fontsize=10)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.show()

    fig, ax = plt.subplots()
    error_contour = ax.contour(xx, yy, error_variance_matrix, 20, extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]), colors="gray")
    ax.clabel(error_contour, inline=True, fontsize=8)
    error_im = ax.imshow(error_variance_matrix, interpolation='none', origin="lower", cmap="RdYlGn_r", extent=(x_est_bounds[0], x_est_bounds[1], y_est_bounds[0], y_est_bounds[1]))
    plt.scatter(x, y, marker="x", color="blue", s=18)
    CB = fig.colorbar(error_im, shrink=0.8)
    CB.set_label("Estimated Error Variance", rotation=270, labelpad=10)
    plt.suptitle("Estimated Error over 2-D Space", fontsize=12)
    plt.title("Nugget = " + str(c0) + " Sill = " + str(c1) + " Range = " + str(a), fontsize=10)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.show()


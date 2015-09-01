import matplotlib
matplotlib.use("agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

def roc_curve(roc_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Probability of False Detection",
              ylabel="Probability of Detection", title="ROC Curve", ticks=np.arange(0, 1.1, 0.1), dpi=300,
              legend_params=dict(loc=4, fontsize=10, framealpha=1, frameon=True)):
    """
    Draws ROC curves for a set of DistributedROC objects.

    :param roc_objs: list or array of DistributedROC Objects.
    :param obj_labels:
    :param colors:
    :param markers:
    :param filename:
    :param figsize:
    :param xlabel:
    :param ylabel:
    :param ticks:
    :param dpi:
    :return:
    """
    plt.figure(figsize=figsize)
    plt.plot(ticks, ticks, "k--")
    for r, roc_obj in enumerate(roc_objs):
        roc_data = roc_obj.roc_curve()
        plt.plot(roc_data["POFD"], roc_data["POD"], marker=markers[r], color=colors[r], label=obj_labels[r])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title(title)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def performance_diagram(roc_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Success Ratio (1-FAR)",
                        ylabel="Probability of Detection", ticks=np.arange(0, 1.1, 0.1), dpi=300, csi_cmap="Blues",
                        csi_label="Critical Success Index", title="Performance Diagram",
                        legend_params=dict(loc=4, fontsize=10, framealpha=1, frameon=True)):
    """
    Draws a performance diagram from a set of DistributedROC objects.

    :param roc_objs: list or array of DistributedROC Objects.
    :param obj_labels: list or array of labels describing each DistributedROC object.
    :param colors: list of color strings
    :param markers: list of markers.
    :param filename: output filename.
    :param figsize: tuple with size of the figure in inches.
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param ticks: Array of ticks used for x and y axes
    :param dpi: DPI of the output image
    :param csi_cmap: Colormap used for the CSI contours
    :param csi_label: Label for the CSI colorbar
    :return:
    """
    plt.figure(figsize=figsize)
    grid_ticks = np.arange(0, 1.01, 0.01)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)
    csi_contour = plt.contourf(sr_g, pod_g, csi, np.arange(0.1, 1.1, 0.1), extend="max", cmap=csi_cmap)
    b_contour = plt.contour(sr_g, pod_g, bias, [0.5, 1, 1.5, 2, 4], colors="k", linestyles="dashed")
    plt.clabel(b_contour, fmt="%1.1f", manual=[(0.2, 0.9), (0.4, 0.9), (0.6, 0.9), (0.7, 0.7)])
    for r, roc_obj in enumerate(roc_objs):
        perf_data = roc_obj.performance_curve()
        plt.plot(1 - perf_data["FAR"], perf_data["POD"], marker=markers[r], color=colors[r], label=obj_labels[r])
    cbar = plt.colorbar(csi_contour)
    cbar.set_label(csi_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title(title)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def reliability_diagram(rel_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Forecast Probability",
                        ylabel="Observed Relative Frequency", ticks=np.arange(0, 1.05, 0.05), dpi=300, inset_size=1.5,
                        legend_params=dict(loc=0, fontsize=10, framealpha=1, frameon=True)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(ticks, ticks, "k--")
    inset_hist = inset_axes(ax, width=inset_size, height=inset_size, loc=2)
    for r, rel_obj in enumerate(rel_objs):
        rel_curve = rel_obj.reliability_curve()
        plt.plot(rel_curve["Bin_Start"], rel_curve["Positive_Relative_Freq"], color=colors[r], marker=markers[r],
                 label=obj_labels[r])
        inset_hist.semilogy(rel_curve["Bin_Start"], rel_curve["Total_Relative_Freq"], color=colors[r],
                            marker=markers[r])
        inset_hist.set_xlabel("Forecast Probability")
        inset_hist.set_ylabel("Forecast Relative Frequency")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def attributes_diagram(rel_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Forecast Probability",
                        ylabel="Observed Relative Frequency", ticks=np.arange(0, 1.05, 0.05), dpi=300, inset_size="20%",
                        title="Attributes Diagram",
                        legend_params=dict(loc=0, fontsize=8, framealpha=1, frameon=True)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(ticks, ticks, "k--")
    inset_hist = inset_axes(ax, width=inset_size, height=inset_size, loc=2)
    climo = rel_objs[0].climatology()
    no_skill = 0.5 * ticks + 0.5 * climo
    skill_x = [climo, climo, 1, 1, climo, climo, 0, 0, climo]
    skill_y = [climo, 1, 1, no_skill[-1], climo, 0, 0, no_skill[0], climo]
    plt.fill(skill_x, skill_y, "0.8")
    plt.plot(ticks, np.ones(ticks.shape) * climo, "k--")
    for r, rel_obj in enumerate(rel_objs):
        rel_curve = rel_obj.reliability_curve()
        ax.plot(rel_curve["Bin_Start"], rel_curve["Positive_Relative_Freq"], color=colors[r], marker=markers[r],
                 label=obj_labels[r])
        inset_hist.semilogy(rel_curve["Bin_Start"], rel_curve["Total_Relative_Freq"], color=colors[r],
                            marker=markers[r])
    inset_hist.set_xlabel("Forecast Probability")
    inset_hist.set_ylabel("Forecast Relative Frequency")
    ax.annotate("No Skill", (0.6, no_skill[6]), rotation=22.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend(**legend_params)
    ax.set_title(title)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()

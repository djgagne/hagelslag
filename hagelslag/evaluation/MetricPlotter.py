import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition


def roc_curve(roc_objs, obj_labels, colors, markers, filename, figsize=(8, 8),
              xlabel="Probability of False Detection",
              ylabel="Probability of Detection",
              title="ROC Curve", ticks=np.arange(0, 1.1, 0.1), dpi=300,
              legend_params=None, bootstrap_sets=None, ci=(2.5, 97.5),
              label_fontsize=14, title_fontsize=16, tick_fontsize=12):
    """
    Plots a set receiver/relative operating characteristic (ROC) curves from DistributedROC objects.

    The ROC curve shows how well a forecast discriminates between two outcomes over a series of thresholds. It
    features Probability of Detection (True Positive Rate) on the y-axis and Probability of False Detection
    (False Alarm Rate) on the x-axis. This plotting function allows one to customize the colors and markers of the
    ROC curves as well as the parameters of the legend and the title.

    Args:
        roc_objs (list): DistributedROC objects being plotted.
        obj_labels (list): Label describing the forecast associated with a DistributedROC object.
        colors (list): List of matplotlib-readable colors (names or hex-values) for each curve.
        markers (list): Matplotlib marker (e.g. *, o, v, etc.) for each curve.
        filename (str): Name of figure file being saved.
        figsize (tuple): (Width, height) of the figure in inches.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): The title of the figure.
        ticks (numpy.ndarray): Values shown on the x and y axes.
        dpi (int): Figure resolution in dots per inch.
        legend_params (None, dict): Keyword arguments for the formatting of the figure legend.
        bootstrap_sets (list): List of lists of DistributedROC objects that were bootstrap resampled for each model.
        ci (tuple of 2 floats): Quantiles of the edges of the bootstrap confidence intervals ranging from 0 to 100.
        label_fontsize (int): Font size of the x and y axis labels.
        title_fontsize (int): Font size of the title.
        tick_fontsize (int): Font size of the x and y tick labels.

    Examples:

        >>> from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
        >>> import numpy as np
        >>> forecasts = np.random.random(1000)
        >>> obs = np.random.random_integers(0, 1, 1000)
        >>> roc = DistributedROC()
        >>> roc.update(forecasts, obs)
        >>> roc_curve([roc], ["Random"], ["orange"], ["o"], "random_roc.png")
    """
    if legend_params is None:
        legend_params = dict(loc=4, fontsize=12, framealpha=1, frameon=True)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(ticks, ticks, "k--")
    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            broc_curves = np.dstack([b_roc.roc_curve().values for b_roc in b_set])
            pod_range = np.percentile(broc_curves[:, 0], ci, axis=1)
            pofd_range = np.percentile(broc_curves[:, 1], ci, axis=1)
            pod_poly = np.concatenate((pod_range[1], pod_range[0, ::-1]))
            pofd_poly = np.concatenate((pofd_range[0], pofd_range[1, ::-1]))
            pod_poly[np.isnan(pod_poly)] = 0
            pofd_poly[np.isnan(pofd_poly)] = 0
            plt.fill(pofd_poly, pod_poly, alpha=0.5, color=colors[b])
    for r, roc_obj in enumerate(roc_objs):
        roc_data = roc_obj.roc_curve()
        plt.plot(roc_data["POFD"], roc_data["POD"], marker=markers[r], color=colors[r], label=obj_labels[r])
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(ticks, fontsize=tick_fontsize)
    plt.yticks(ticks, fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def performance_diagram(roc_objs, obj_labels, colors, markers, filename, figsize=(8, 8),
                        xlabel="Success Ratio (1-FAR)",
                        ylabel="Probability of Detection", ticks=np.arange(0, 1.1, 0.1),
                        dpi=300, csi_cmap="Blues",
                        csi_label="Critical Success Index", title="Performance Diagram",
                        legend_params=None, bootstrap_sets=None, ci=(2.5, 97.5), label_fontsize=14,
                        title_fontsize=16, tick_fontsize=12):
    """
    Draws a performance diagram from a set of DistributedROC objects.

    A performance diagram is a variation on the ROC curve in which the Probability of False Detection on the
    x-axis has been replaced with the Success Ratio (1-False Alarm Ratio or Precision). The diagram also shows
    the Critical Success Index (CSI or Threat Score) as a series of curved contours, and the frequency bias as
    angled diagonal lines. Points along the 1:1 diagonal are unbiased, and better performing models should appear
    in the upper right corner. The performance diagram is particularly useful for displaying verification for
    severe weather warnings as it displays all three commonly used statistics (POD, FAR, and CSI) simultaneously
    on the same chart.

    Args:
        roc_objs (list): DistributedROC objects being plotted.
        obj_labels: list or array of labels describing each DistributedROC object.
        obj_labels (list): Label describing the forecast associated with a DistributedROC object.
        colors (list): List of matplotlib-readable colors (names or hex-values) for each curve.
        markers (list): Matplotlib marker (e.g. *, o, v, etc.) for each curve.
        filename (str): Name of figure file being saved.
        figsize (tuple): (Width, height) of the figure in inches.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): The title of the figure.
        ticks (numpy.ndarray): Values shown on the x and y axes.
        dpi (int): Figure resolution in dots per inch.
        csi_cmap (str): Matplotlib colormap used to fill CSI contours.
        csi_label (str): Label for CSI colormap.
        legend_params (None or dict): Keyword arguments for the formatting of the figure legend.
        bootstrap_sets (list): A list of arrays of bootstrapped DistributedROC objects. If not None,
            confidence regions will be plotted.
        ci (tuple): tuple of bootstrap confidence interval percentiles.
        label_fontsize (int): Font size of the x and y axis labels.
        title_fontsize (int): Font size of the title.
        tick_fontsize (int): Font size of the x and y tick labels.

    Examples:

        >>> from hagelslag.evaluation.ProbabilityMetrics import DistributedROC
        >>> import numpy as np
        >>> forecasts = np.random.random(1000)
        >>> obs = np.random.random_integers(0, 1, 1000)
        >>> roc = DistributedROC()
        >>> roc.update(forecasts, obs)
        >>> performance_diagram([roc], ["Random"], ["orange"], ["o"], "random_performance.png")
    """
    if legend_params is None:
        legend_params = dict(loc=4, fontsize=10, framealpha=1, frameon=True)
    plt.figure(figsize=figsize)
    grid_ticks = np.arange(0, 1.01, 0.01)
    sr_g, pod_g = np.meshgrid(grid_ticks, grid_ticks)
    bias = pod_g / sr_g
    csi = 1.0 / (1.0 / sr_g + 1.0 / pod_g - 1.0)
    csi_contour = plt.contourf(sr_g, pod_g, csi, np.arange(0.1, 1.1, 0.1), extend="max", cmap=csi_cmap)
    b_contour = plt.contour(sr_g, pod_g, bias, [0.5, 1, 1.5, 2, 4], colors="k", linestyles="dashed")
    plt.clabel(b_contour, fmt="%1.1f", manual=[(0.2, 0.9), (0.4, 0.9), (0.6, 0.9), (0.7, 0.7)])
    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            perf_curves = np.dstack([b_roc.performance_curve().values for b_roc in b_set])
            pod_range = np.nanpercentile(perf_curves[:, 0], ci, axis=1)
            sr_range = np.nanpercentile(1 - perf_curves[:, 1], ci, axis=1)
            pod_poly = np.concatenate((pod_range[1], pod_range[0, ::-1]))
            sr_poly = np.concatenate((sr_range[1], sr_range[0, ::-1]))
            pod_poly[np.isnan(pod_poly)] = 0
            sr_poly[np.isnan(sr_poly)] = 1
            plt.fill(sr_poly, pod_poly, alpha=0.5, color=colors[b])
    for r, roc_obj in enumerate(roc_objs):
        perf_data = roc_obj.performance_curve()
        plt.plot(1 - perf_data["FAR"], perf_data["POD"], marker=markers[r], color=colors[r], label=obj_labels[r])
    cbar = plt.colorbar(csi_contour)
    cbar.set_label(csi_label)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(ticks, fontsize=tick_fontsize)
    plt.yticks(ticks, fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(**legend_params)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def reliability_diagram(rel_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Forecast Probability",
                        ylabel="Observed Relative Frequency", ticks=np.arange(0, 1.05, 0.05), dpi=300, inset_size=1.5,
                        title="Reliability Diagram", legend_params=None, bootstrap_sets=None, ci=(2.5, 97.5)):
    """
    Plot reliability curves against a 1:1 diagonal to determine if probability forecasts are consistent with their
    observed relative frequency.

    Args:
        rel_objs (list): List of DistributedReliability objects.
        obj_labels (list): List of labels describing the forecast model associated with each curve.
        colors (list): List of colors for each line
        markers (list): List of line markers
        filename (str): Where to save the figure.
        figsize (tuple): (Width, height) of the figure in inches.
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        ticks (array): Tick value labels for the x and y axes.
        dpi (int): resolution of the saved figure in dots per inch.
        inset_size (float): Size of inset
        title (str): Title of figure
        legend_params (dict): Keyword arguments for the plot legend.
        bootstrap_sets (list): A list of arrays of bootstrapped DistributedROC objects. If not None,
            confidence regions will be plotted.
        ci (tuple): tuple of bootstrap confidence interval percentiles
    """
    if legend_params is None:
        legend_params = dict(loc=4, fontsize=10, framealpha=1, frameon=True)
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(ticks, ticks, "k--")
    inset_hist = inset_axes(ax, width=inset_size, height=inset_size, loc=2)
    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            brel_curves = np.dstack([b_rel.reliability_curve().values for b_rel in b_set])
            bin_range = np.percentile(brel_curves[:, 0], ci, axis=1)
            rel_range = np.percentile(brel_curves[:, 3], ci, axis=1)
            bin_poly = np.concatenate((bin_range[1], bin_range[0, ::-1]))
            rel_poly = np.concatenate((rel_range[1], rel_range[0, ::-1]))
            bin_poly[np.isnan(bin_poly)] = 0
            rel_poly[np.isnan(rel_poly)] = 0
            plt.fill(bin_poly, rel_poly, alpha=0.5, color=colors[b])
    for r, rel_obj in enumerate(rel_objs):
        rel_curve = rel_obj.reliability_curve()
        ax.plot(rel_curve["Bin_Start"], rel_curve["Positive_Relative_Freq"], color=colors[r], marker=markers[r],
                label=obj_labels[r])
        inset_hist.semilogy(rel_curve["Bin_Start"], rel_curve["Total_Relative_Freq"], color=colors[r],
                            marker=markers[r])
        inset_hist.set_xlabel("Forecast Probability")
        inset_hist.set_ylabel("Forecast Relative Frequency")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.legend(**legend_params)
    ax.set_title(title)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def attributes_diagram(rel_objs, obj_labels, colors, markers, filename, figsize=(8, 8), xlabel="Forecast Probability",
                       ylabel="Observed Relative Frequency", ticks=np.arange(0, 1.05, 0.05), dpi=300, 
                       title="Attributes Diagram", legend_params=None, inset_params=None,
                       inset_position=(0.12, 0.72, 0.25, 0.25), bootstrap_sets=None, ci=(2.5, 97.5)):
    """
    Plot reliability curves against a 1:1 diagonal to determine if probability forecasts are consistent with their
    observed relative frequency. Also adds gray areas to show where the climatological probabilities lie and what
    areas result in a positive Brier Skill Score.

    Args:
        rel_objs (list): List of DistributedReliability objects.
        obj_labels (list): List of labels describing the forecast model associated with each curve.
        colors (list): List of colors for each line
        markers (list): List of line markers
        filename (str): Where to save the figure.
        figsize (tuple): (Width, height) of the figure in inches.
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        ticks (`numpy.ndarray`): Tick value labels for the x and y axes.
        dpi (int): resolution of the saved figure in dots per inch.
        title (str): Title of figure
        legend_params (dict): Keyword arguments for the plot legend.
        inset_params (dict): Keyword arguments for the inset axis.
        inset_position (tuple): Position of the inset axis in normalized axes coordinates (left, bottom, width, height)
        bootstrap_sets (list): A list of arrays of bootstrapped DistributedROC objects. If not None,
            confidence regions will be plotted.
        ci (tuple): tuple of bootstrap confidence interval percentiles
    """
    if legend_params is None:
        legend_params = dict(loc=4, fontsize=10, framealpha=1, frameon=True)
    if inset_params is None:
        inset_params = dict(width="25%", height="25%", loc=2, axes_kwargs=dict(facecolor='white'))
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(ticks, ticks, "k--")
    inset_hist = inset_axes(ax, **inset_params)
    ip = InsetPosition(ax, inset_position)
    inset_hist.set_axes_locator(ip)
    climo = rel_objs[0].climatology()
    no_skill = 0.5 * ticks + 0.5 * climo
    skill_x = [climo, climo, 1, 1, climo, climo, 0, 0, climo]
    skill_y = [climo, 1, 1, no_skill[-1], climo, 0, 0, no_skill[0], climo]
    f = ax.fill(skill_x, skill_y, "0.8")
    f[0].set_zorder(1)
    ax.plot(ticks, np.ones(ticks.shape) * climo, "k--")
    if bootstrap_sets is not None:
        for b, b_set in enumerate(bootstrap_sets):
            brel_curves = np.vstack([b_rel.reliability_curve()["Positive_Relative_Freq"].values for b_rel in b_set])
            rel_range = np.nanpercentile(brel_curves, ci, axis=0)
            fb = ax.fill_between(b_set[0].thresholds[:-1], rel_range[1], rel_range[0], alpha=0.5, color=colors[b])
            fb.set_zorder(2)
    for r, rel_obj in enumerate(rel_objs):
        rel_curve = rel_obj.reliability_curve()
        ax.plot(rel_curve["Bin_Start"], rel_curve["Positive_Relative_Freq"], color=colors[r],
                marker=markers[r], label=obj_labels[r])
        inset_hist.semilogy(rel_curve["Bin_Start"] * 100, rel_obj.frequencies["Total_Freq"][:-1], color=colors[r],
                            marker=markers[r])
    inset_hist.set_xlabel("Forecast Probability")
    inset_hist.set_ylabel("Frequency")
    ax.annotate("No Skill", (0.6, no_skill[12]), rotation=22.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    ax.set_xticklabels((ticks * 100).astype(int))
    ax.set_yticks(ticks)
    ax.set_yticklabels((ticks * 100).astype(int))
    ax.legend(**legend_params)
    ax.set_title(title)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()

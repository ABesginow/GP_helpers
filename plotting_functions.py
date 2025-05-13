import matplotlib.pyplot as plt
import numpy as np



# Usage examples:
#plot_training_data(train_x, train_y)
#plot_training_data(train_x, train_y, plot_separately=True, ncols=2, titles="Sensor")
#plot_training_data(train_x, train_y, colors=['red', 'green', 'blue'], show=True)
def plot_training_data(
    X_train,
    Y_train,
    show=True,
    return_fig=False,
    plot_separately=False,
    fig=None,
    ax=None,
    colors=None,
    ncols=1,
    figsize=None,
    titles=None,
    xlabel="Input",
    ylabel="Output",
):
    """
    Plot raw training data with optional per-channel subplots or combined plot.

    Parameters:
    - X_train: (N,) input values
    - Y_train: (N,) or (N, D) output values
    - show: whether to call plt.show()
    - return_fig: whether to return figure
    - plot_separately: if True, use subplots; otherwise, combine in one plot
    - fig, ax: optionally provide existing matplotlib Figure and Axes
    - colors: optional list of colors per output
    - ncols: number of columns for subplot layout
    - figsize: figure size tuple
    - titles: optional list of titles or base title string
    - xlabel: label for x-axis
    - ylabel: label for y-axis
    
    Returns:
    - fig (optional): the matplotlib Figure object if return_fig is True
    """

    X_train = np.asarray(X_train).squeeze()
    Y_train = np.asarray(Y_train)

    if Y_train.ndim == 1:
        Y_train = Y_train[:, np.newaxis]

    num_outputs = Y_train.shape[1]

    if colors is None:
        colors = plt.cm.tab10.colors
    if len(colors) < num_outputs:
        colors = (colors * ((num_outputs // len(colors)) + 1))[:num_outputs]

    if plot_separately:
        nrows = int(np.ceil(num_outputs / ncols))
        if ax is None or fig is None:
            fig, ax = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 3 * nrows), squeeze=False)
        ax = ax.flatten()

        if isinstance(titles, str):
            titles = [f"{titles} {i}" for i in range(num_outputs)]
        elif titles is None:
            titles = [f"Output Dimension {i}" for i in range(num_outputs)]

        for i in range(num_outputs):
            ax_i = ax[i]
            ax_i.plot(X_train, Y_train[:, i], '.', color=colors[i], label=f"Channel {i}")
            ax_i.set_title(titles[i])
            ax_i.set_xlabel(xlabel)
            ax_i.set_ylabel(ylabel)
            ax_i.grid(True)

        for j in range(num_outputs, len(ax)):
            fig.delaxes(ax[j])

        plt.tight_layout()

    else:
        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=figsize or (8, 6))

        for i in range(num_outputs):
            ax.plot(X_train, Y_train[:, i], '.', color=colors[i], label=f"Channel {i}")
        ax.set_title(titles if isinstance(titles, str) else "Training Data")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    if show:
        plt.show()

    if return_fig:
        return fig


# Usage examples:
# plot_gp_predictions(..., ncols=2, titles=["Pressure", "Temperature"], xlabel="Time")
# colors = ['red', 'green', 'blue']
def plot_single_input_gp_posterior(
    X_train, Y_train,
    X_test, Y_pred_mean, Y_pred_var,
    n_std=2,
    show=True,
    return_fig=False,
    fig=None,
    ax=None,
    colors=None,
    ncols=1,
    figsize=None,
    titles=None,
    xlabel="Input",
    ylabel="Output"
):
    """
    Plot predictions from a single- or multi-output Gaussian Process model.

    Parameters:
    - X_train: (N,) training inputs
    - Y_train: (N,) or (N, D) training outputs
    - X_test: (M,) test inputs
    - Y_pred_mean: (M,) or (M, D) predicted mean at test inputs
    - Y_pred_var: (M,) or (M, D) predicted variance at test inputs
    - n_std: credible interval size (default: 2)
    - show: whether to call plt.show()
    - return_fig: whether to return fig
    - fig, ax: external matplotlib figure/axes
    - colors: optional list of colors for different outputs
    - ncols: number of columns in subplot layout
    - figsize: figure size tuple
    - titles: list of subplot titles or single base title string
    - xlabel: x-axis label (shared or per-plot)
    - ylabel: y-axis label (shared or per-plot)
    
    Returns:
    - fig if return_fig is True
    """

    # Input normalization
    X_train = np.asarray(X_train).squeeze()
    X_test = np.asarray(X_test).squeeze()
    Y_train = np.asarray(Y_train)
    Y_pred_mean = np.asarray(Y_pred_mean)
    Y_pred_var = np.asarray(Y_pred_var)

    if Y_train.ndim == 1:
        Y_train = Y_train[:, np.newaxis]
    if Y_pred_mean.ndim == 1:
        Y_pred_mean = Y_pred_mean[:, np.newaxis]
    if Y_pred_var.ndim == 1:
        Y_pred_var = Y_pred_var[:, np.newaxis]

    num_outputs = Y_train.shape[1]

    # Handle colors
    if colors is None:
        colors = plt.cm.tab10.colors  # default color cycle
    if len(colors) < num_outputs:
        # a // b  is the same as floor(a / b)
        colors = (colors * ((num_outputs // len(colors)) + 1))[:num_outputs]

    # Handle subplot layout
    nrows = int(np.ceil(num_outputs / ncols))
    if ax is None or fig is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 3 * nrows), squeeze=False)
    ax = ax.flatten()

    # Handle titles
    if isinstance(titles, str):
        titles = [f"{titles} {i}" for i in range(num_outputs)]
    elif titles is None:
        titles = [f"Output Dimension {i}" for i in range(num_outputs)]

    for i in range(num_outputs):
        ax_i = ax[i]

        color = colors[i]

        # Training data
        ax_i.scatter(X_train, Y_train[:, i], color='black', s=20, label='Training Data', zorder=3, marker='x')

        # Posterior mean
        ax_i.plot(X_test, Y_pred_mean[:, i], color=color, label='Posterior Mean')

        # Credible interval
        std_dev = np.sqrt(Y_pred_var[:, i])
        lower = Y_pred_mean[:, i] - n_std * std_dev
        upper = Y_pred_mean[:, i] + n_std * std_dev
        ax_i.fill_between(X_test, lower, upper, color=color, alpha=0.3, label=f'{n_std}$\sigma$ Interval')

        # Titles and labels
        ax_i.set_title(titles[i])
        ax_i.set_xlabel(xlabel)
        ax_i.set_ylabel(ylabel)
        ax_i.legend()
        ax_i.grid(True)

    # Hide unused subplots if any
    for j in range(num_outputs, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()

    if show:
        plt.show()

    if return_fig:
        return fig

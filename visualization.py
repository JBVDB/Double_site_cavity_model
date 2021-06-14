import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from typing import Union
import Bio.PDB


def plot_training_history_v2(
    model_name: str,
    folder:str="models/double_cav_models/",
    return_figs=False):
    """
    Plot the results of the training with the metrics saved in a pickle file 
    with naming scheme '{folder}/metrics_{model_name}.pickle'.
    """
    with open(f"{folder}/metrics_{model_name}.pickle", "rb") as f:
        history = pickle.load(f)

    history.pop("best_epoch")

    history_loss = {}
    for mode in history:
        for key in history[mode]:
            if "loss" in key:
                history_loss[mode] = history[mode][key]
        history[mode].pop("loss")

    # Plot loss evolution:
    fig = plt.figure(figsize=(5, 4), constrained_layout=True)

    for mode in history_loss:
        n_records = len(history_loss[mode])
        plt.plot(range(0, n_records), history_loss[mode], label=mode)

    plt.xlabel("Epoch")
    plt.xticks(range(0, n_records+1, 2))
    plt.ylabel("Cross-Entropy Loss")
    legend = plt.legend(shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.grid(linestyle="-", alpha=0.5)

    fig_name = f"results/double_cav_models/training_model_{model_name}_loss.png"
    plt.savefig(fig_name, dpi=200, bbox_inches = "tight")
    plt.show()

    # Plot accuracy evolution:
    fig2 = plt.figure(figsize=(6.6, 4), constrained_layout=True)

    max_y = history["train"][max(history["train"], key=history["train"].get)]
    palette_iter = iter(sns.color_palette("bright", as_cmap=True))
    for mode, palette in zip(history, ["YlOrRd", "PuBuGn"]):
        train_colors = plt.get_cmap("YlOrRd")

        plt.plot(range(0, n_records), history[mode][f"acc_join"],
                label=f"{mode} (R1, R2)", color=next(palette_iter))
        plt.plot(range(0, n_records), history[mode][f"acc_res1"],
                label=f"{mode} R1", color=next(palette_iter))
        plt.plot(range(0, n_records), history[mode][f"acc_res2"],
                label=f"{mode} R2", color=next(palette_iter))
        plt.plot(range(0, n_records), history[mode][f"acc_res2_given_res1"],
                label=f"{mode} R2 | R1", color=next(palette_iter), linestyle="--")
        plt.plot(range(0, n_records), history[mode][f"acc_res1_given_res2"],
                label=f"{mode} R1 | R2", color=next(palette_iter), linestyle="--")

    plt.xlabel("Epoch")
    plt.xticks(range(0, n_records+1, 2))
    plt.yticks(np.arange(0., max(max_y)+0.05, 0.05))

    plt.ylabel("Accuracy")
    legend = plt.legend(shadow=True,
                    loc='upper right',
                    bbox_to_anchor=(1.35, 1.02),
                    )
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.grid(linestyle="-", alpha=0.5)

    fig2_name = f"results/double_cav_models/training_model_{model_name}_acc.png"
    plt.savefig(fig2_name, dpi=200, bbox_inches = "tight")
    plt.show()

    if return_figs:
        return (fig, fig_name, fig2, fig2_name)


def plot_training_history(folder="models/double_cav_models/",
                            model_name: Union[str, int]=0):
    """Plot training results, for both loss and accuracies."""
    print("Beware, this is the new version (epoch 0 must be the initialization.")
    with open(f"models/double_cav_models/metrics_{model_name}.pickle", "rb") as f:
        rec = pickle.load(f)
    # Plot loss evolution.
    plt.figure(figsize=(5, 4), constrained_layout=True)
    n_records = len(rec[f"loss_train"])
    plt.plot(range(0, n_records), rec[f"loss_train"], label="train")
    plt.plot(range(0, n_records), rec[f"loss_val"], label="val")
    plt.xlabel("Epoch")
    plt.xticks(range(0, n_records+1, 2))
    plt.ylabel("Cross-Entropy Loss")
    legend = plt.legend(shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.grid(linestyle="-", alpha=0.5)
    plt.savefig(f"results/double_cav_models/training_model_{model_name}_loss.png",
                dpi=200, bbox_inches = "tight")
    plt.show()

    # Plot accuracies' evolution.
    plt.figure(figsize=(6.6, 4), constrained_layout=True)

    rec.pop("best_epoch")
    rec.pop("loss_train")
    rec.pop("loss_val")
    max_y = rec[max(rec, key=rec.get)]
    train_colors = plt.get_cmap("YlOrRd")
    val_colors = plt.get_cmap("PuBuGn")
    palette_iter = iter(sns.color_palette("bright", as_cmap=True))

    plt.plot(range(0, n_records), rec[f"acc_train_join"],
             label="Train (R1, R2)", color=next(palette_iter))
    plt.plot(range(0, n_records), rec[f"acc_train_res1"],
             label="Train R1", color=next(palette_iter))
    plt.plot(range(0, n_records), rec[f"acc_train_res2"],
             label="Train R2", color=next(palette_iter))
    plt.plot(range(0, n_records), rec["acc_train_res2_given_res1"],
             label="Train R2 | R1", color=next(palette_iter), linestyle="--")
    plt.plot(range(0, n_records), rec["acc_train_res1_given_res2"],
             label="Train R1 | R2", color=next(palette_iter), linestyle="--")

    plt.plot(range(0, n_records), rec[f"acc_val_join"],
             label="Val (R1, R2)", color=next(palette_iter))
    plt.plot(range(0, n_records), rec[f"acc_val_res1"],
             label="Val R1", color=next(palette_iter))
    plt.plot(range(0, n_records), rec[f"acc_val_res2"],
             label="Val R2", color=next(palette_iter))
    plt.plot(range(0, n_records), rec[f"acc_val_res2_given_res1"],
             label="Val R2 | R1", color=next(palette_iter), linestyle="--")
    plt.plot(range(0, n_records), rec[f"acc_val_res1_given_res2"],
             label="Val R1 | R2", color=next(palette_iter), linestyle="--")

    plt.xlabel("Epoch")
    plt.xticks(range(0, n_records+1, 2))
    plt.yticks(np.arange(0., max(max_y)+0.05, 0.05))
    plt.ylabel("Accuracy")
    legend = plt.legend(shadow=True,
                    loc='upper right',
                    bbox_to_anchor=(1.35, 1.02),
                    )
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.grid(linestyle="-", alpha=0.5)
    plt.savefig(f"results/double_cav_models/training_model_{model_name}_acc.png",
                dpi=200, bbox_inches = "tight")
    plt.show()


def show_respair_acc_heatmap(array_pairres_acc: np.array,
                             model_name: str,
                             keep_order=True,
                             ):
    """Plot heat map of accuracies of pairs of residue type."""

    get_index_to_one = np.vectorize(
        lambda x: Bio.PDB.Polypeptide.index_to_one(x)
        )

    plt.figure(figsize=(10, 9))

    if not keep_order:
        mask = np.zeros_like(array_pairres_acc)
        mask[np.triu_indices_from(mask, k=1)] = True
        kwargs = {"mask": mask}
        suffix = ""
        array_pairres_acc = array_pairres_acc.T

    else:
        kwargs = {}
        # suffix = "_order_kept"

    ax = sns.heatmap(array_pairres_acc,
            vmax=np.max(array_pairres_acc),
            square=True,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            linewidth = 1.5,
            cbar=False,
            xticklabels=get_index_to_one(np.arange(0, 20)),
            yticklabels=get_index_to_one(np.arange(0, 20)),
            **kwargs
            )

    plt.yticks(rotation=0)

    # Create a divider of the existing ax.
    divider = make_axes_locatable(ax)

    # Append a new ax on it on the right.
    cax = divider.append_axes("right", size=0.25, pad=0.2)
    cb = plt.colorbar(ax.get_children()[0], cax=cax)

    plt.savefig(
        f"results/double_cav_models/heatmap_{model_name}.png",
        dpi=200,
        bbox_inches = "tight"
                )
    plt.show()
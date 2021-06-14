# To log AFTER run!
import neptune.new as neptune
from neptune.new.types import File # interactive plots
from neptune.sessions import Session

import inspect

import pickle
import torch

# import plotly
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.graph_objects as go

from visualization import(
    plot_training_history_v2,
    )


def set_and_save_metadata(dict_metadata: dict,
    RESUME: bool,
    MODELS_DIRPATH="models/double_cav_models/"
    ):
    """Infer remaining metadata, show current model set up."""

    print(f"BEWARE THAT RESUME == {RESUME}!\n\n")
    print(f"Batch size: {dict_metadata['batch_size']}")

    print(f"model: {dict_metadata['model_name']}, mode resume={RESUME},"
    f" for {dict_metadata['epoch']} epochs.")
    model_archi = dict_metadata["cav_model"]("cpu")
    print()
    print(model_archi)
    print()

    # Check and set train_val_split.
    with open(dict_metadata["data_path"], "rb") as f_train:
        parsed_pdb_filenames = pickle.load(f_train)

    TRAIN_VAL_SPLIT = round(
        dict_metadata["n_train"]/len(parsed_pdb_filenames), 4)

    assert dict_metadata["n_train"] == int(
        len(parsed_pdb_filenames)*TRAIN_VAL_SPLIT)

    dict_metadata['n_val'] = int(round(
        (1-TRAIN_VAL_SPLIT)*len(parsed_pdb_filenames)))

    print(
        f"TRAIN VAL SPLIT: {TRAIN_VAL_SPLIT}, "
        f"train_set: {dict_metadata['n_train']}, "
        f"val_set: {dict_metadata['n_val']}")

    dict_metadata["train_val_split"] = TRAIN_VAL_SPLIT

    # TO CHANGE!!!! use tags to investigate (either list or str, use new stuff from mcoding)
    # Check if weights.
    if dict_metadata["weight"] == "":
        dict_metadata["weight"] = None
    else:
        pass # to implement

    # Check if matrix factorization:
    if dict_metadata["matfact_k"] != "" and dict_metadata["matfact_k"] is not None:
        assert dict_metadata["matfact_k"]*40 == dict_metadata["output_shape"]

    # Check if cuda
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        # print(torch.cuda.get_device_properties("cuda"))

    # Set remaining metadata and pickle it
    with open(
        f"{MODELS_DIRPATH}/{dict_metadata['model_name']}_metadata.pickle",
        "wb") as f:
        pickle.dump(dict_metadata, f)

    return dict_metadata, parsed_pdb_filenames


def log_metadata_to_neptune(meta: dict,
    run_name="",
    project_name="thesis",
    MODELS_DIRPATH="models/double_cav_models"
    ):
    """
    Log all metadata, metrics evolution, models parameters, plots to Neptune.
    """
    # API_TOKEN = getpass("Enter Neptune Api_token: ")
    API_TOKEN = (
        "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaH"
        "R0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNzEyY2Q4MS02ZjUwLTQ4ZGUt"
        "OWQ2YS0xZWEwNmU5ZjRkZjAifQ==")

    run = neptune.init(project=f'jbvdb/{project_name}',
                    api_token=API_TOKEN,
                    run=run_name)

    # for key in meta:
    #     run[f"_/{key}"] = meta[key]  

    run["parameters"] = meta
    run["parameters/grid_size"] = [
        meta["grid_dim_xy"],
        meta["grid_dim_xy"],
        meta["grid_dim_z"]
        ]
    run["sys/tags"].add(meta['tag'])

    with open("temp_file.txt", "w") as f:
        f.writelines(inspect.getsource(meta["cav_model"]))

    run["models/architecture"].upload("temp_file.txt")
    run["models/metadata"].upload(
        f"{MODELS_DIRPATH}/{meta['model_name']}_metadata.pickle")

    # Save and log training history.
    with open(
        f"{MODELS_DIRPATH}/metrics_{meta['model_name']}.pickle", "rb"
        ) as f:
        history = pickle.load(f)

    best_epoch = history.pop("best_epoch")
    best_epoch_res = {"train": {}, "val": {}}
    for mode in history:
        for metric in history[mode]:
            for i, epoch in enumerate(history[mode][metric]):
                run[f"{mode}/{metric}"].log(history[mode][metric][i])
                if i == best_epoch:
                    best_epoch_res[mode][metric] = round(
                                                    history[mode][metric][i], 4)

    run["val_loss"] = history["val"]["loss"][best_epoch]
    run["val_acc_join"] = history["val"]["acc_join"][best_epoch]
    run["models/history"].upload(
        f"{MODELS_DIRPATH}/metrics_{meta['model_name']}.pickle")
    run["models/best_epoch"] = best_epoch
    run["models/best_epoch_results"] = best_epoch_res

    # Save model parameters, 
    run["models/best_model"].upload(
        f"{MODELS_DIRPATH}/{meta['model_name']}_epoch_{best_epoch}.pt")
    print(f"Successfully uploaded model {best_epoch}")

    # Save matplotlib and plotly plots.
    (fig, fig_name, fig2, fig2_name) = plot_training_history_v2(
        meta['model_name'],
        return_figs=True)


    run["plots/loss"].upload(fig_name)
    run["plots/acc"].upload(fig2_name)
    run["plots/loss_i"] = File.as_html(convert_plt_to_plotly(fig))
    run["plots/acc_i"] = File.as_html(convert_plt_to_plotly(fig2))

    run.stop()
    print("Model successfully logged to Neptune. Run logging stopped.")


def convert_plt_to_plotly(fig):
    ax_list = fig.axes
    for ax in ax_list:
        ax.get_legend().remove()

    plotly_fig = tls.mpl_to_plotly(fig)
    legend = go.layout.Legend(
        x=1.0,
        y=1.0,
        bordercolor="black"
    )
    plotly_fig.update_layout(showlegend=True, legend=legend)
    return plotly_fig
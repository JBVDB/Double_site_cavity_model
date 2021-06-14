from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import torch
import timeit
import pickle
import glob
from torch.utils.data import DataLoader, Dataset

# from cavity_model import (
#     CavityModel,
#     ResidueEnvironmentsDataset,
#     ToTensor,
#     DDGDataset,
#     DDGToTensor,
#     DownstreamModel,
# )


def _train(
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    cavity_model_net: CavityModel,
    loss_function: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    EPOCHS: int,
    PATIENCE_CUTOFF: int,
    matfact_k: int,
    output_shape: int,
    folder="models/double_cav_models",
    model_name="model",
    resume=False,
    ):
    """
    Helper function to perform training loop for the Cavity Model.
    """

    current_best_epoch = -1
    curr_best_metric = 1e4
    patience = 0 # we start at 0
    current_epoch = -1

    early_stop_metric = "loss"
    if loss_function.weight is not None: # no class imbalance correction
        early_stop_metric = "acc_join"
        curr_best_metric = -1

    # Resume training
    if len(glob.glob(f"{folder}/{model_name}_epoch_*.pt")) > 0:
        epochs_saved = [int(x.split("_epoch_")[1][:-3]) for x in glob.glob(
            f"{folder}/{model_name}*")]

        if resume:
            current_epoch = max(epochs_saved)
            model_path = f"{folder}/{model_name}_epoch_{current_epoch}.pt"
            checkpoint = torch.load(model_path)

            assert checkpoint["epoch"] == current_epoch, (
            f"checkpoint epoch {checkpoint['epoch']}",
            f"does not match current epoch {current_epoch}"
            )

            print(f"Training resumed from epoch {current_epoch}.\n")

            current_best_epoch = checkpoint["current_best_epoch"]
            curr_best_metric = checkpoint[f"current_best_{early_stop_metric}"]
            patience = checkpoint["patience"]

            cavity_model_net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f"Current best epoch: {current_best_epoch}, "
                  f"{early_stop_metric}: {curr_best_metric:5.3f}, "
                  f"Patience: {patience}.")
            print()
        else:
            raise ValueError(f"Epoch file(s) already exist(s) for {model_name}!")

    print(
        f"- Starts training with '{early_stop_metric}' "
        f"as early stop metric...\n")
    
    # Create dict of rec to save.
    rec = dict()

    # Run model.
    for epoch in range(current_epoch+1, EPOCHS+1): # EPOCHS+1 since 0 == ini state
        t1 = timeit.default_timer()

        # Assess model's initial state.
        if epoch == 0:
            print("Evaluating randomly initialized model.")
            loss_train, rec["train"] =  _eval_loop(cavity_model_net,
                                                dataloader_train,
                                                loss_function,
                                                matfact_k,
                                                output_shape,
                                                )
        # Train over train batches.
        else:
            loss_train, rec["train"] = _train_loop(cavity_model_net,
                                                dataloader_train,
                                                optimizer,
                                                loss_function,
                                                matfact_k,
                                                output_shape,
                                                )

        rec["train"]["loss"] = loss_train

        print(f"{'train'.upper():5s} - Loss: {rec['train']['loss']:5.3f}, "
        f"Join acc: {rec['train']['acc_join']:5.3f},  "
        f"Res1: {rec['train']['acc_res1']:4.2f}, "
        f"Res2: {rec['train']['acc_res2']:4.2f},  "
        f"Cond 2|1: {rec['train']['acc_res2_given_res1']:4.2f}, "
        f"Cond 1|2: {rec['train']['acc_res1_given_res2']:4.2f}."
        )

        # Validate over val batches.
        loss_val, rec["val"] = _eval_loop(cavity_model_net,
                                          dataloader_val,
                                          loss_function,
                                          matfact_k,
                                          output_shape,
                                          )
        rec["val"]["loss"] = loss_val

        # Show epoch result
        print(f"{'val'.upper():5s} - Loss: {rec['val']['loss']:5.3f}, "
        f"Join acc: {rec['val']['acc_join']:5.3f},  "
        f"Res1: {rec['val']['acc_res1']:4.2f}, "
        f"Res2: {rec['val']['acc_res2']:4.2f},  "
        f"Cond 2|1: {rec['val']['acc_res2_given_res1']:4.2f}, "
        f"Cond 1|2: {rec['val']['acc_res1_given_res2']:4.2f}."
        )

        if early_stop_metric == "acc_join":
            if round(
                rec["val"][early_stop_metric] - curr_best_metric, 5) > 0.001:
                curr_best_metric = rec["val"][early_stop_metric]
                current_best_epoch = epoch
                patience = 0
            else:
                patience += 1
        else:
            if (
                round(
                curr_best_metric - rec["val"][early_stop_metric], 5) > 0.001 or
                epoch == 0):

                curr_best_metric = rec["val"][early_stop_metric]
                current_best_epoch = epoch
                patience = 0
            else:
                patience += 1

        print(
            f"Epoch {epoch:2d} done in {round(timeit.default_timer() - t1, 2)} "
            f"sec.  Patience: {patience}")
        print()

        # Save training states for future resuming.
        state = {
            "epoch": epoch,
            "model_state_dict": cavity_model_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "patience": patience,
            f"current_best_{early_stop_metric}": curr_best_metric,
            "current_best_epoch": current_best_epoch,
        }
        model_path = f"{folder}/{model_name}_epoch_{epoch}.pt"

        torch.save(state, model_path)

        # Keep track rec (SAME ORDER as names_rec_to_save)
        rec_path = f"{folder}/metrics_{model_name}"

        if epoch > 0:
            with open(f"{rec_path}.pickle", "rb") as handle:
                history = pickle.load(handle)
            for key in rec:
                for metric in rec[key]:
                    history[key][metric].append(rec[key][metric])
            history["best_epoch"] = current_best_epoch
        else:
            history = dict()
            for key in rec:
                history[key] = dict()
                for metric in rec[key]:
                    history[key][metric] = [rec[key][metric]]

        # Pickle rec + best model name.        
        with open(f"{rec_path}.pickle", "wb") as handle:
            pickle.dump(history, handle)

        # Assess Early stopping.
        if patience > PATIENCE_CUTOFF:
            print("Early stopping activated.")
            break

    best_model_path = f"{folder}/{model_name}_epoch_{current_best_epoch}.pt"
    print(
        f"Best epoch idx: {current_best_epoch} with validation {early_stop_metric}: "
        f"{curr_best_metric:5.3f}\nFound at: "
        f"'{best_model_path}'"
    )

    return best_model_path


def _train_loop(
    cavity_model_net: CavityModel,
    dataloader_train: DataLoader,
    optimizer: torch.optim.Adam,
    loss_function: torch.nn.CrossEntropyLoss,
    matfact_k: int,
    output_shape: int,
    ) -> (torch.Tensor, float):
    """
    Helper function to perform a train loop
    """
    labels_true = []
    labels_pred = []
    loss_batch_list = []

    idx_res_split = output_shape // 2

    cavity_model_net.train()
    for batch_x, batch_y in tqdm.tqdm(dataloader_train,
                                      total=len(dataloader_train),
                                      unit="batch",
                                    ):
        optimizer.zero_grad()

        batch_y_pred = cavity_model_net(batch_x)

        # Split predictions in (20, k) x (20, k) for matrix factorization
        batch_y_pred_res1 = batch_y_pred[:, :idx_res_split].reshape(
            -1, 20, matfact_k)
        batch_y_pred_res2 = batch_y_pred[:, idx_res_split:].reshape(
            -1, matfact_k, 20)

        batch_y_pred = (batch_y_pred_res1 @ batch_y_pred_res2).reshape(-1, 400)

        loss_batch = loss_function(batch_y_pred, torch.argmax(batch_y, dim=1))
        loss_batch.backward()
        optimizer.step()

        loss_batch_list.append(loss_batch.detach().cpu().item())

        # Save joint, conditional, marginal, restype accuracies.
        labels_true.append(
            np.vstack(np.unravel_index(
                torch.argmax(batch_y, dim=1).detach().cpu().numpy(),
                (20, 20)
                )).T
        )

        labels_pred.append(
            np.vstack(np.unravel_index(
                torch.argmax(batch_y_pred, dim=1).detach().cpu().numpy(),
                (20, 20)
                )).T
        )

    loss_train = np.mean(loss_batch_list)

    return (loss_train, _get_accuracies(
                                    labels_true,
                                    labels_pred)
    )


def _eval_loop(
    cavity_model_net: CavityModel,
    dataloader_val: DataLoader,
    loss_function: torch.nn.CrossEntropyLoss,
    matfact_k: int,
    output_shape: int,
    **kwargs
    ) -> Tuple:
    """
    Helper function to perform an eval loop
    """
    # Eval loop. Due to memory, we don't pass the whole eval set to the model

    labels_true = []
    labels_pred = []
    loss_batch_list = []

    idx_res_split = output_shape // 2

    cavity_model_net.eval()
    with torch.set_grad_enabled(False):
        for batch_x, batch_y in tqdm.tqdm(dataloader_val,
                                        total=len(dataloader_val),
                                        unit="batch",
                                        leave=False
                                        ):
            batch_y_pred = cavity_model_net(batch_x)

            # Split predictions in (20, k) x (20, k) for matrix factorization
            batch_y_pred_res1 = batch_y_pred[:, :idx_res_split].reshape(
                -1, 20, matfact_k)
            batch_y_pred_res2 = batch_y_pred[:, idx_res_split:].reshape(
                -1, matfact_k, 20)

            batch_y_pred = (batch_y_pred_res1 @ batch_y_pred_res2).reshape(
                -1, 400)

            loss_batch = loss_function(
                batch_y_pred, torch.argmax(batch_y, dim=1))
            loss_batch_list.append(loss_batch.detach().cpu().item())

            # Save joint, conditional, marginal, restype accuracies.
            labels_true.append(
                np.vstack(np.unravel_index(
                    torch.argmax(batch_y, dim=1).detach().cpu().numpy(),
                    (20, 20)
                    )).T
            )

            labels_pred.append(
                np.vstack(np.unravel_index(
                    torch.argmax(batch_y_pred, dim=1).detach().cpu().numpy(),
                    (20, 20)
                    )).T
            )

    loss_val = np.mean(loss_batch_list)

    # return (loss_val, labels_true, labels_pred)
    return (loss_val, _get_accuracies( # Unpack tuple of accuracies.
                                labels_true,
                                labels_pred,
                                **kwargs)
    )


def _get_accuracies(labels_true: List[int],
                    labels_pred: List[int],
                    get_restypes_acc=False,
                    keep_pair_order=True,
    ):
    """ 
    compute join, marginal, conditional and restype accuracies
    from lists of true and predicted labels.
    """
    rec = dict()

    # Create arrays from lists.
    # labels_true = np.reshape(labels_true, (-1, 2))
    labels_true = np.vstack(labels_true)
    # labels_pred = np.reshape(labels_pred, (-1, 2))
    labels_pred = np.vstack(labels_pred)

    # joint accuracy
    mask_join = np.logical_and(labels_true[:, 0] == labels_pred[:, 0],
                               labels_true[:, 1] == labels_pred[:, 1])

    rec["acc_join"] = np.mean(mask_join)

    # marginal accuracies

    mask_res1_true = (labels_true[:, 0] == labels_pred[:, 0])
    mask_res2_true = (labels_true[:, 1] == labels_pred[:, 1])

    rec["acc_res1"] = np.mean(mask_res1_true)
    rec["acc_res2"] = np.mean(mask_res2_true)

    # conditional accuracies

    mask_res1_true = mask_res1_true.nonzero()
    mask_res2_true = mask_res2_true.nonzero()
    
    rec["acc_res2_given_res1"] = np.mean(
        labels_true[mask_res1_true, 1] == labels_pred[mask_res1_true, 1]
        )
    rec["acc_res1_given_res2"] = np.mean(
        labels_true[mask_res2_true, 0] == labels_pred[mask_res2_true, 0]
        )

    if get_restypes_acc:
        # save restypes accuracies
        pairres_count_true = np.zeros((20, 20), dtype=np.int)
        pairres_count = np.zeros((20, 20), dtype=np.int)
        pairres_count_true = np.zeros((20, 20), dtype=np.int)
        pairres_count = np.zeros((20, 20), dtype=np.int)

        # Get accuracies per residue type.
        mask_join = mask_join.nonzero()
        
        if keep_pair_order:
        # if order matters (can retrieve easily marginals):
            mask_pairres_count_true, count_true = np.unique(
                labels_pred[mask_join],
                return_counts=True, axis=0
                )
            mask_pairres_count, count = np.unique(
                labels_pred, return_counts=True, axis=0
                )

        else:
            # if order does not matter:
            mask_pairres_count_true, count_true = np.unique(
                np.sort(labels_pred[mask_join], axis=1),
                return_counts=True, axis=0
                )
            mask_pairres_count, count = np.unique(
                np.sort(labels_pred, axis=1),
                return_counts=True, axis=0
                )

        pairres_count_true[mask_pairres_count_true[:, 0],
                        mask_pairres_count_true[:, 1]] += count_true
        pairres_count[mask_pairres_count[:, 0],
                    mask_pairres_count[:, 1]] += count

        pairres_count[pairres_count == 0] = 1 # avoid 0 division.

        rec["pairres_count_true"] = pairres_count_true
        rec["pairres_count"] = pairres_count
    else:
        pass

    return rec


def _train_val_split(
    parsed_pdb_filenames: List[str],
    TRAIN_VAL_SPLIT: float,
    DEVICE: str,
    BATCH_SIZE: int,
    **kwargs
    ):
    """
    Helper function to perform training and validation split of ResidueEnvironments. Note that
    we do the split on PDB level not on ResidueEnvironment level due to possible leakage.
    """
    n_train_pdbs = int(len(parsed_pdb_filenames) * TRAIN_VAL_SPLIT)
    filenames_train = parsed_pdb_filenames[:n_train_pdbs]
    filenames_val = parsed_pdb_filenames[n_train_pdbs:]

    to_tensor_transformer = ToTensor(DEVICE, **kwargs) # allow for unravel indexing

    dataset_train = ResidueEnvironmentsDataset(
        filenames_train, transformer=to_tensor_transformer # thanks to call function
    )


    dataloader_train = DataLoader( # read the data (and shuffle it) within batch size and put into memory.
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=to_tensor_transformer.collate_cat, # avoid having to load data to CUDA in the NN model itself!
        # collate_fn=to_tensor_transformer.collate_wrapper,
        drop_last=True, # drop_last=True parameter ignores the last batch (when the number of examples in your dataset is not divisible by your batch_size
        # pin_memory=True
    )

    print(
        f"Training data set includes {len(filenames_train)} pdbs with "
        f"{len(dataset_train)} environments."
    )

    # dataset_train = 0

    dataset_val = ResidueEnvironmentsDataset(
        filenames_val, transformer=to_tensor_transformer
    )

    # TODO: Fix it so drop_last doesn't have to be True when calculating validation accuracy.
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=to_tensor_transformer.collate_cat, # if using /batch, callable that specifies how the batch is created.
        # collate_fn=to_tensor_transformer.collate_wrapper,
        drop_last=True, # ignores the last batch (when the number of examples in your dataset is not divisible by your batch_size
        # pin_memory=True
    )

    print(
        f"Validation data set includes {len(filenames_val)} pdbs with "
        f"{len(dataset_val)} environments."
    )

    # dataset_val = 0

    return dataloader_train, dataset_train, dataloader_val, dataset_val


def get_test_dataloader(
    test_filenames: List[str],
    BATCH_SIZE: int,
    DEVICE: str,
    reshape_index=True,
    unravel_index=True,    
    ):
    """Return a dataloder for testing."""
    to_tensor_transformer = ToTensor(DEVICE,
                                     unravel_index=unravel_index,
                                     reshape_index=reshape_index)

    dataset_test = ResidueEnvironmentsDataset(
        test_filenames,
        transformer=to_tensor_transformer
        )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=to_tensor_transformer.collate_cat,
        drop_last=False,
        )

    print(
        f"Testing data set includes {len(test_filenames)} pdbs with "
        f"{len(dataset_test)} environments."
    )    

    return dataset_test, dataloader_test


def _test(
    cavity_model_net: CavityModel,
    dataloader_test: DataLoader,
    loss_function: torch.nn.CrossEntropyLoss,
    matfact_k: int,
    output_shape: int,
    get_restypes_acc=True,
    keep_pair_order=True,
):
    return _eval_loop(cavity_model_net,
                      dataloader_test,
                      loss_function,
                      matfact_k,
                      output_shape,
                      get_restypes_acc=get_restypes_acc,
                      keep_pair_order=keep_pair_order,
                      )


def _predict(
    cavity_model_net: CavityModel,
    dataloader_infer: DataLoader,
    matfact_k: int,
    output_shape: int,
    ):
    """
    Get predicted proba distribution per pair_res environment.
    Made for making prediction one protein at a time, returning an array
    (n_pairs, 400) long.
    """

    labels_true = []
    idx_res_split = output_shape // 2

    softmax = torch.nn.Softmax(dim=1)
    cavity_model_net.eval()
    with torch.set_grad_enabled(False):
        for batch_x, batch_y in dataloader_infer:

            batch_y_pred = cavity_model_net(batch_x)

            # Split predictions in (20, k) x (20, k) for matrix factorization
            batch_y_pred_res1 = batch_y_pred[:, :idx_res_split].reshape(
                -1, 20, matfact_k)
            batch_y_pred_res2 = batch_y_pred[:, idx_res_split:].reshape(
                -1, matfact_k, 20)

            batch_y_pred = (batch_y_pred_res1 @ batch_y_pred_res2).reshape(
                -1, 400)
            batch_y_pred = softmax(batch_y_pred).detach().cpu().numpy()

            # Save true labels
            labels_true.append(
                np.vstack(np.unravel_index(
                    torch.argmax(batch_y, dim=1).detach().cpu().numpy(),
                    (20, 20)
                    )).T
            )

    return batch_y_pred, labels_true


def get_best_epoch_perf(model_name: str= "",
                        models_dirpath="models/double_cav_models/"):
    """
    Fetch training history of model_name,
    return results of the best epoch as string.
    """
    with open(f"{models_dirpath}/metrics_{model_name}.pickle", "rb") as f:
        history = pickle.load(f)
    best_epoch = history.pop("best_epoch")
    best_epoch_perf = {}
    for key in history:
        best_epoch_perf[key] = {}
        for metric in history[key]:
            best_epoch_perf[key][metric] = history[key][metric][best_epoch]

    best_epoch_perf = f"Best epoch: {best_epoch}\n"\
    f"{'train'.upper():5s} - "\
    f"Loss: {best_epoch_perf['train']['loss']:5.3f}, "\
    f"Join acc: {best_epoch_perf['train']['acc_join']:5.3f},  "\
    f"Res1: {best_epoch_perf['train']['acc_res1']:4.2f}, "\
    f"Res2: {best_epoch_perf['train']['acc_res2']:4.2f},  "\
    f"Cond 2|1: {best_epoch_perf['train']['acc_res2_given_res1']:4.2f}, "\
    f"Cond 1|2: {best_epoch_perf['train']['acc_res1_given_res2']:4.2f}."\
    f"\n"\
    f"{'val'.upper():5s} - Loss: {best_epoch_perf['val']['loss']:5.3f}, "\
    f"Join acc: {best_epoch_perf['val']['acc_join']:5.3f},  "\
    f"Res1: {best_epoch_perf['val']['acc_res1']:4.2f}, "\
    f"Res2: {best_epoch_perf['val']['acc_res2']:4.2f},  "\
    f"Cond 2|1: {best_epoch_perf['val']['acc_res2_given_res1']:4.2f}, "\
    f"Cond 1|2: {best_epoch_perf['val']['acc_res1_given_res2']:4.2f}."

    return best_epoch_perf


# Tools for saving model summary (objective: combine the 2!)
def get_df_summary(text: str):
    from torchsummary import summary
    text = text.split("\n")

    def parse_line(line: list):
        parsed_line = []
        for el in line:
            if not el == "":
                parsed_line.append(el)
        return parsed_line

    keys = ["Layer (type)", "Output shape", "Param #"]
    df_summary = {k: [] for k in keys}

    for line in text:
        line = parse_line(line.split("  "))
        df_summary["Layer (type)"].append(line[0])
        df_summary["Output shape"].append(line[1])
        df_summary["Param #"].append(line[2])
    return pd.DataFrame(df_summary)


def get_and_save_model_summary(model: CavityModel,
                               input_size: tuple,
                               model_name: "cavity_model"):
    import io
    from torchsummary import summary
    from contextlib import redirect_stdout

    # Context manager for temporarily redirecting sys.stdout to another file or file-like object.
    with open(f'models/{model_name}_summary.txt', 'w') as f:
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model=model, input_size=input_size)
        out = f.getvalue()
        return out


# Tools for sending run's completion notification
def test_login_smtp_server(
    sender_email = "",
    receiver_email = "",
    ):
    """
    In case of error 534: # https://accounts.google.com/DisplayUnlockCaptcha
    Debug ref: https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
    """
    port = 587  # For starttls (tls encryption protocol)
    smtp_server = "smtp.gmail.com"

    password = getpass("Type password: ")

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        # TODO: Send email here
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()
    return password


def send_run_results(
    h: dict,
    password: str,
    models_dirpath="models/double_cav_models/",
    sender_email="",
    receiver_email="",
    ):

    port = 587  # For starttls (tls encryption protocol)
    smtp_server = "smtp.gmail.com"

    subject_email = f"run of Model: {h['model_name']} completed."

    perf_best_epoch = get_best_epoch_perf(h["model_name"],
                                          models_dirpath=models_dirpath)

    hyperparam_records = f"""Hyperparameters:
    ----------------
    """
    for key in h:
        if key != "model_name":
            hyperparam_records += f"{key}: {h[key]}\n"
    
    message = """\
    From: {}
    To: {}
    Subject: {}

    {}

    {}""".format(
        sender_email,
        receiver_email,
        subject_email,
        perf_best_epoch,
        hyperparam_records
        )
    message = "\n".join([line.lstrip() for line in message.split("\n")])

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

        print(f"Email sent to {receiver_email}.")

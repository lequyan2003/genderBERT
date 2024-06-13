import datetime
import json
import logging
import math
import random
import sys

import torch

from model import (
    create_dataloader,
    create_model,
    create_optimizer,
    eval_model,
    load_embeddings,
    load_to_cuda,
    test_model,
    train_epoch,
)


def load_config(config_path, mode):
    """
    Load and set configuration from a JSON file based on the mode.

    Args:
        config_path (str): Path to the configuration file.
        mode (str): Mode specifying the configuration to use.

    Returns:
        dict: Configuration parameters for the specified mode.
    """
    with open(config_path, "r") as fp:
        config = json.load(fp)
    if mode not in config:
        raise ValueError(f"Invalid mode {mode} in config.")
    return config[mode]


def set_seed(seed_val=42):
    """
    Set the seed for reproducibility.

    Args:
        seed_val (int, optional): Seed value. Defaults to 42.
    """
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def main(return_model: bool):
    """
    Main function of the module. Create a model, train and evaluate it based
    on the config JSON.
    Steps:
    1. Create/Load embeddings for train, validation, and test data.
    2. Create dataloader from embeddings for faster and easier training.
    3. Create model, optimizer, and scheduler. Load to CUDA.
    4. Train, validate, and test the model (according to config).
    5. Save/return model if parameter is set.

    Args:
        return_model (bool): If True, return the trained model and statistics;
        otherwise, return only statistics.

    Returns:
        If return_model is True, returns tuple(model, stats). Otherwise,
        returns stats (dict).
    """
    if len(sys.argv) != 2:
        logging.error("Invalid number of arguments!")
        return

    mode = sys.argv[1]
    config = load_config("config.json", mode)

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
        level=logging.INFO,
    )

    log_starttime = datetime.datetime.now()

    # Set seed for reproducibility
    set_seed()

    # Load embeddings
    train_data, val_data, test_data = load_embeddings(
        config["DATASET_TYPE"],
        config["PATH_TRAIN"],
        config["COLUMNS_TRAIN"],
        config["PATH_VALIDATION"],
        config["COLUMNS_VALIDATION"],
        config["PATH_TEST"],
        config["COLUMNS_TEST"],
        config["MODEL_TYPE"],
        config["TOGGLE_PHASES"],
        config["LOAD_EMBEDDINGS"],
        config["ROWS_COUNTS"],
        config["MAX_TOKENCOUNT"],
        config["TRUNCATING_METHOD"],
        save_embeddings=False,
    )
    logging.info("Creating dataloaders ...")
    train_dataloader = create_dataloader(train_data, config["BATCH_SIZE"])
    val_dataloader = create_dataloader(val_data, config["BATCH_SIZE"])
    test_dataloader = create_dataloader(test_data, 256)

    # Create model
    model = create_model(
        config["PRELOAD_MODEL"], config["MODEL_TYPE"], config["BASE_FREEZE"]
    )

    # Set usage of GPU or CPU
    device = load_to_cuda(model)

    if config["TOGGLE_PHASES"][0]:
        num_train_steps = (
            math.ceil(len(train_dataloader) / config["BATCH_SIZE"])
            * config["EPOCHS"]
        )
        optimizer, scheduler = create_optimizer(
            model, config["LEARNING_RATE"], num_train_steps
        )

    # Train model
    stats = {}
    for epoch_i in range(config["EPOCHS"]):
        logging.info(
            f"-------- Epoch {epoch_i + 1} / {config['EPOCHS']} --------"
        )
        if config["TOGGLE_PHASES"][0]:
            loss, acc = train_epoch(
                model, train_dataloader, optimizer, device, scheduler
            )
            stats[str(epoch_i)] = {"loss": loss, "accuracy": acc}
        # Validation
        if config["TOGGLE_PHASES"][1]:
            acc = eval_model(
                model, val_dataloader, device, config["MODEL_TYPE"]
            )
    # Testing
    if config["TOGGLE_PHASES"][2]:
        acc = test_model(model, test_dataloader, device, config["MODEL_TYPE"])

    # Saving the model
    if config["SAVE_MODEL"]:
        logging.info("Saving the model...")
        model.save_pretrained(config["SAVE_MODEL"])

    logging.info("Training complete!")
    log_endtime = datetime.datetime.now()
    log_runtime = log_endtime - log_starttime
    logging.info("Total runtime: " + str(log_runtime))

    if return_model:
        return model, stats
    else:
        return stats


if __name__ == "__main__":
    main(False)

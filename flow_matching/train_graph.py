import utils
import torch
import hydra
import models
import training_state

from omegaconf import OmegaConf, open_dict
import common
import os
import time

import warnings
# filter out warnings Functor warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config_graph")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(sample_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

    # Preparing dataset
    train_set, val_set = utils.load_graph_data(cfg.task, augment = cfg.augment, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = train_set[0][0].shape
    dataloader = utils.GraphDataLoader(train_set, val_set, cfg.batch_size, cfg.val_batch_size, device)
    with open_dict(cfg):
        if cfg.family in ["flow_matching", "imf"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })

        else:
            raise NotImplementedError

    # Preparing model, optimizer, and grad scaler (for AMP)
    if cfg.implementation == "custom":
        model = models.get_model_class(cfg.family)(**cfg.model).to(device)
    else:
        raise NotImplementedError
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    grad_scaler = torch.cuda.amp.GradScaler(
        enabled=(device == "cuda" and cfg.family == "flow_matching")
    )
    train_metrics = common.Metrics()

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists
    print(OmegaConf.to_yaml(cfg))
    print(f"model has {num_params} params")
    if cfg.family == "imf":
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_manifest = training_state.build_data_manifest(cfg.task, repo_root)
        run_metadata = training_state.initialize_run_artifacts(
            log_dir, cfg, data_manifest, repo_root
        )
        training_state.register_training_state(
            checkpointer,
            model,
            optim,
            step,
            grad_scaler,
            cfg.family,
            run_metadata["config_hash"],
            run_metadata["data_fingerprint"],
            training_state.resolved_config_yaml(cfg),
        )
        restore_mode, restore_path = training_state.restore_training_state(
            checkpointer,
            cfg.family,
            resume=cfg.resume,
            init_from=cfg.init_from,
            legacy_from_checkpoint=cfg.from_checkpoint,
            device=device,
        )
        training_state.record_restore(log_dir, restore_mode, restore_path)
    else:
        load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler)

    # Start training
    print(f"==== Start Training on Device: {device} ====")
    model.train()

    t_0 = time.time()
    t_1 = time.time()
    best_loss = 1e12
    while step < cfg.train_steps:
        x, cond, _ = dataloader.get_batch("train")
        # x has (B, N, 2); netlist_data is a single graph in tg.Data format
        t = torch.randint(
            1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device=device
        ) if cfg.family == "flow_matching" else None
        optim.zero_grad()

        loss, model_metrics = model.loss(x, cond, t)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optim)
        grad_scaler.update()

        train_metrics.add({"loss": loss.detach().cpu().item()})
        train_metrics.add(model_metrics)
        logger.add(train_metrics.result())
        step.increment()
        if cfg.family == "imf" and (
                int(step) == 1 or int(step) % cfg.print_every == 0):
            training_state.write_metrics(log_dir, {
                "step": int(step),
                **model_metrics,
            })

        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            x_val, cond_val, _ = dataloader.get_batch("val")
            train_logs = utils.validate(x, model, cond)
            val_logs = utils.validate(x_val, model, cond_val)

            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            #logger.add(train_metrics.result())
            logger.add(val_logs, prefix="val")
            logger.add(train_logs, prefix="train")

            # display example images
            for split in ["train", "val"]:
                x_disp, cond_disp = dataloader.get_display_batch(cfg.display_examples, split = split)
                utils.display_graph_samples(cfg.display_examples, x_disp, cond_disp, model, logger, prefix = split)
                utils.display_forward_graph_samples(x_disp, cond_disp, model, logger, prefix = split)
            logger.write()
            t_1 = t_2

            save_checkpoint(checkpointer, cfg, log_dir) # save latest checkpoint
            if val_logs["loss"] < best_loss:
                best_loss = val_logs["loss"]
                save_checkpoint(
                    checkpointer, cfg, log_dir, os.path.join(log_dir, "best.ckpt")
                )
                print("saving best model")
            cond_val.to(device="cpu")

        if (cfg.eval_every > 0) and (int(step)) % cfg.eval_every == 0 or int(step) == 1:
            print(f"saving model at step {int(step)}")
            save_checkpoint(
                checkpointer,
                cfg,
                log_dir,
                os.path.join(log_dir, f"step_{int(step)}.ckpt"),
            )
            print("generating evaluation report")
            t3 = time.time()
            utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy = cfg.eval_policy)
            logger.write()
            t4 = time.time()
            print(f"generated report in {t4-t3:.3f} sec")
        

        cond.to(device="cpu")

def load_checkpoint(checkpointer, cfg, step, model, optim, grad_scaler):
    checkpointer.register({
            "step": step,
            "model": model,
            "optim": optim,
            "grad_scaler": grad_scaler,
        })
    if cfg.mode == "train":
        checkpointer.load(
            path_override = None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
            else os.path.join(cfg.log_dir, cfg.from_checkpoint)
        )
    elif cfg.mode in ["finetune", "ddpo"]:
        # Try to resume existing run
        loaded = checkpointer.load()
        if not loaded:
            # No existing run, so load pre-trained model only
            loaded = checkpointer.load(
                path_override = os.path.join(cfg.log_dir, cfg.from_checkpoint),
                filter_keys = ["model"],
            )
            if not loaded:
                print("WARNING Failed to load checkpoint for finetuning. Training from scratch instead.")
    else:
        raise NotImplementedError


def save_checkpoint(checkpointer, cfg, log_dir, path=None):
    if cfg.family == "imf":
        return training_state.save_training_checkpoint(
            checkpointer,
            path or checkpointer.ckpt_path,
            log_dir,
        )
    checkpointer.save(path)
    return None

if __name__=="__main__":
    main()

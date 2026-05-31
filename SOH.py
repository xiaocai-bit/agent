import torch
from torch import nn, optim
from types import SimpleNamespace
from pathlib import Path
import json
import csv
import random
import numpy as np

from MIT_loader import MITDdataset

from models import (
    TSLLMModel,
    CrossTemporalSOHModel,
    PhysicsNet
)

from train_utils import evaluate, evaluate_with_outputs
from prompts import build_dataloaders


class HistoricalCycleDataset(torch.utils.data.Dataset):
    """Adds a causal K-cycle history window to a cycle-level TensorDataset.

    Each base sample is still a 67-dimensional current-cycle representation.
    The added history tensors provide ``X_{t-K+1:t}`` and their normalized cycle
    coordinates so the model can learn a historical degradation representation.
    """

    def __init__(self, base_dataset, history_window):
        self.base_dataset = base_dataset
        self.history_window = int(history_window)

        if hasattr(base_dataset, "tensors") and len(base_dataset.tensors) >= 3:
            features, _, cycle_id = base_dataset.tensors[:3]
        else:
            samples = [base_dataset[i] for i in range(len(base_dataset))]
            if not samples or len(samples[0]) < 3:
                raise TypeError(
                    "Cross-temporal SOH training expects samples with "
                    "(features, labels, cycle_id)."
                )
            features = torch.stack([sample[0] for sample in samples], dim=0)
            cycle_id = torch.stack([sample[2] for sample in samples], dim=0)
        cycles = cycle_id.reshape(cycle_id.size(0), -1)[:, 0]
        sorted_indices = torch.argsort(cycles, stable=True)

        history_indices = torch.empty(
            (len(base_dataset), self.history_window),
            dtype=torch.long
        )

        for sorted_pos, sample_idx in enumerate(sorted_indices.tolist()):
            start = max(0, sorted_pos - self.history_window + 1)
            window = sorted_indices[start:sorted_pos + 1]
            if window.numel() < self.history_window:
                pad = window[0].repeat(self.history_window - window.numel())
                window = torch.cat([pad, window], dim=0)
            history_indices[sample_idx] = window

        self.history_features = features[history_indices].clone()
        self.history_cycle_id = cycle_id[history_indices].clone()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, cycle_id = self.base_dataset[idx]
        return (
            x,
            y,
            cycle_id,
            self.history_features[idx],
            self.history_cycle_id[idx]
        )


def add_history_to_loader(loader, history_window):
    history_dataset = HistoricalCycleDataset(loader.dataset, history_window)
    return torch.utils.data.DataLoader(
        history_dataset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
        drop_last=loader.drop_last,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False)
    )


# =========================================================
# CONFIG
# =========================================================
SOH_CONFIG = {

    # =====================================================
    # data
    # =====================================================
    "data": "MIT",

    "input_type": "handcraft_features",

    "batch": 2,

    "batch_size": 4,

    "normalized_type": "minmax",

    "minmax_range": (0, 1),

    # =====================================================
    # LLM
    # =====================================================
    "llm_path": r"D:\Code_LTF\model_fine_turing\large_model\qwen\Qwen3-4B",

    # =====================================================
    # training
    # =====================================================
    "epochs": 50,

    "lr": 1e-4,
    'use_prompt':False,

    "weight_decay": 1e-4,

    "dropout": 0.1,

    "physics_lambda": 0.1,

    "monotonic_lambda": 0.005,

    "use_cross_temporal": True,

    "history_window": 10,

    "history_lambda": 0.5,

    "cross_temporal_lambda": 0.05,

    # =====================================================
    # misc
    # =====================================================
    "random_seed": 2026,

    "seed": 2023,

    "test_battery_id": 1,
}


# =========================================================
# RUNNER
# =========================================================
def run_soh(args, device):

    cfg = SimpleNamespace(**SOH_CONFIG)

    # use one unified seed from CLI for all modules
    cfg.seed = int(getattr(args, "seed", cfg.seed))
    cfg.random_seed = cfg.seed

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg.device = device

    # =====================================================
    # evaluate wrapper
    # =====================================================
    def evaluate_with_optional_preprocess(loader):

        eval_kwargs = {
            "model": model,
            "device": device,
            "loader": loader,
            "loss_fn": loss_fn
        }

        return evaluate(**eval_kwargs)

    # =====================================================
    # dataset
    # =====================================================
    loader = MITDdataset(cfg)

    data_dict = loader.get_features(
        test_battery_id=cfg.test_battery_id
    )

    # =====================================================
    # dataloader
    # must return:
    # (x, y, cycle_id)
    # =====================================================
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        data_dict
    )

    if cfg.use_cross_temporal:
        train_loader = add_history_to_loader(
            train_loader,
            cfg.history_window
        )
        val_loader = add_history_to_loader(
            val_loader,
            cfg.history_window
        )
        test_loader = add_history_to_loader(
            test_loader,
            cfg.history_window
        )

    # =====================================================
    # model
    # latent token PINN backbone
    # =====================================================
    backbone = TSLLMModel(

        input_dim=67,

        num_tokens=4,

        token_dim=128,

        llm_path=cfg.llm_path,

        dropout=cfg.dropout,

        device=device,

        llm_dtype="bf16"

    )

    if cfg.use_cross_temporal:
        model = CrossTemporalSOHModel(
            backbone=backbone,
            history_len=cfg.history_window,
            dropout=cfg.dropout
        ).to(device)
    else:
        model = backbone.to(device)

    # =====================================================
    # physics net
    # =====================================================
    physics_net = PhysicsNet(

        latent_dim=model.llm_enc.hidden_size

    ).to(device)

    if cfg.use_cross_temporal:
        print("\n[Model] TSLLMModel + CrossTemporalSOHModel + PhysicsNet")
    else:
        print("\n[Model] TSLLMModel + PhysicsNet")

    # =====================================================
    # optimizer
    # =====================================================
    optimizer = optim.AdamW(

        list(model.parameters())
        + list(physics_net.parameters()),

        lr=cfg.lr,

        weight_decay=cfg.weight_decay
    )

    # =====================================================
    # scheduler
    # =====================================================
    scheduler = optim.lr_scheduler.MultiStepLR(

        optimizer,

        milestones=[30, 70],

        gamma=0.5
    )

    # =====================================================
    # loss
    # =====================================================
    loss_fn = nn.MSELoss()

    # =====================================================
    # initialize
    # =====================================================
    best_val = 1e10

    best_epoch = 0

    best_state = None
    train_loss_history = []
    val_loss_history = []

    # =====================================================
    # training
    # =====================================================
    print("\n========================")
    print("Start Training")
    print("========================")

    for epoch in range(cfg.epochs):

        model.train()

        physics_net.train()

        train_loss_sum = 0.0

        # =================================================
        # train loop
        # =================================================
        for batch in train_loader:

            # =============================================
            # batch
            # =============================================
            if len(batch) == 5:
                ts_x, y, cycle_id, history_x, history_cycle_id = batch
                history_x = history_x.float().to(device)
                history_cycle = history_cycle_id.float().to(device)
            else:
                ts_x, y, cycle_id = batch
                history_x = None
                history_cycle = None

            ts_x = ts_x.float().to(device)

            y = y.float().to(device)

            cycle_norm = cycle_id.float().to(device).requires_grad_(True)

            # =============================================
            # forward
            # =============================================
            if history_x is not None and cfg.use_cross_temporal:
                outputs = model(
                    ts_x,
                    cycle_norm,
                    history_features=history_x,
                    history_cycle_id=history_cycle,
                    return_cross_temporal=True
                )
                pred = outputs["pred_current"].unsqueeze(-1)
                latent_h = outputs["z_current"]
                pred_history = outputs["pred_history"].unsqueeze(-1)
                z_hat_current = outputs["z_hat_current"]
            else:
                pred, latent_h = model(

                    ts_x,

                    cycle_norm,

                    return_latent=True
                )

                pred = pred.unsqueeze(-1)
                pred_history = None
                z_hat_current = None

            # =============================================
            # data loss
            # =============================================
            data_loss = loss_fn(
                pred,
                y
            )

            # =============================================
            # dSOH/dN
            # =============================================
            dsoh_dn = torch.autograd.grad(

                outputs=pred.sum(),

                inputs=cycle_norm,

                create_graph=True,

                retain_graph=True

            )[0]

            # =============================================
            # physics rhs
            # F(h,SOH,N)
            # =============================================
            physics_rhs = physics_net(

                latent_h,

                pred,

                cycle_norm
            )

            # =============================================
            # physics loss
            # =============================================
            physics_loss = (

                (dsoh_dn.squeeze(-1) - physics_rhs) ** 2

            ).mean()

            # =============================================
            # monotonic degradation
            # dSOH/dN <= 0
            # =============================================
            monotonic_loss = torch.relu(
                dsoh_dn.squeeze(-1)
            ).mean()

            # =============================================
            # cross-temporal losses
            # =============================================
            if pred_history is not None:
                history_loss = loss_fn(
                    pred_history,
                    y
                )
                cross_temporal_loss = loss_fn(
                    z_hat_current.float(),
                    latent_h.float()
                )
            else:
                history_loss = torch.zeros((), device=device)
                cross_temporal_loss = torch.zeros((), device=device)

            # =============================================
            # total loss
            # =============================================
            loss = (

                data_loss

                + cfg.history_lambda * history_loss

                + cfg.cross_temporal_lambda * cross_temporal_loss

                + cfg.physics_lambda * physics_loss

                + cfg.monotonic_lambda * monotonic_loss
            )

            # =============================================
            # backward
            # =============================================
            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss_sum += loss.item()

        # =================================================
        # scheduler
        # =================================================
        scheduler.step()

        # =================================================
        # train loss
        # =================================================
        train_loss = train_loss_sum / len(train_loader)
        train_loss_history.append(float(train_loss))

        # =================================================
        # validation
        # =================================================
        val_loss, metrics = evaluate_with_optional_preprocess(
            val_loader
        )
        val_loss_history.append(float(val_loss))

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        # =================================================
        # logging
        # =================================================
        print(

            f"[SOH] "

            f"epoch=[{epoch + 1}/{cfg.epochs}] "

            f"train_loss={train_loss:.6f} "

            f"val_loss={val_loss:.6f} "

            f"lr={lr:.6f}"
        )

        # =================================================
        # best model
        # =================================================
        if val_loss < best_val:

            best_val = val_loss

            best_epoch = epoch + 1

            best_metrics = metrics

            best_state = {

                "model": model.state_dict(),

                "physics_net": physics_net.state_dict()
            }

            print(

                f"[Best] "

                f"epoch={best_epoch} "

                f"val_loss={best_val:.6f}"
            )

            print(

                f"MAE={metrics['MAE']:.4f} | "

                f"MAPE={metrics['MAPE']:.2f}% | "

                f"RMSE={metrics['RMSE']:.4f} | "

                f"R2={metrics['R2']:.4f}"
            )

    # =====================================================
    # load best
    # =====================================================
    print("\n========================")
    print("Load Best Model")
    print("========================")

    model.load_state_dict(
        best_state["model"]
    )

    physics_net.load_state_dict(
        best_state["physics_net"]
    )

    print(

        f"Best epoch = {best_epoch} | "

        f"Best val_loss = {best_val:.6f}"
    )

    # =====================================================
    # final test
    # =====================================================
    print("\n========================")
    print("FINAL TEST RESULT")
    print("========================")

    test_loss, metrics, y_pred, y_true = evaluate_with_outputs(
        model=model,
        device=device,
        loader=test_loader,
        loss_fn=loss_fn
    )

    print(f"test_loss = {test_loss:.6f}")

    print(f"MAE  = {metrics['MAE']:.4f}")

    print(f"MAPE = {metrics['MAPE']:.2f}%")

    print(f"RMSE = {metrics['RMSE']:.4f}")

    print(f"R2   = {metrics['R2']:.4f}")

    run_name = (
        f"SOH-{cfg.data}-batch{cfg.batch}-test_battery{cfg.test_battery_id}"
    )
    output_root = Path(getattr(args, "result_root", "results"))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "task": "SOH",
        "dataset": cfg.data,
        "batch": cfg.batch,
        "test_battery_id": cfg.test_battery_id,
        "random_seed": cfg.random_seed,
        "seed": cfg.seed,
        "physics_lambda": cfg.physics_lambda,
        "monotonic_lambda": cfg.monotonic_lambda,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "metrics": metrics,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (run_dir / "loss_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_loss_history, val_loss_history), start=1):
            writer.writerow([i, tr, va])

    with (run_dir / "test_pred_true.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "pred", "true"])
        for idx, (pred, true) in enumerate(zip(y_pred.tolist(), y_true.tolist())):
            writer.writerow([idx, pred, true])

    print(f"\n[Save] 本次训练结果已保存到: {run_dir}")

    print("\n========================")
    print("Training Finished")
    print("========================")

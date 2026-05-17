import torch
from torch import nn, optim

from types import SimpleNamespace

from MIT_loader import MITDdataset

from models import (
    LSTMEncoder,
    TSLLMModel
)

from train_utils import evaluate
from prompts import build_dataloaders


# =========================================================
# CONFIG
# =========================================================
SOH_CONFIG = {

    # =====================================================
    # data
    # =====================================================
    "data": "MIT",

    "input_type": "partial_charge",

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
    "epochs": 100,

    "lr": 1e-4,

    "weight_decay": 1e-4,

    "dropout": 0.1,
    "use_prompt": False,
    # =====================================================
    # misc
    # =====================================================
    "random_seed": 2023,

    "seed": 2023,

    "test_battery_id": 1,
}


# =========================================================
# RUNNER
# =========================================================
def run_soh(args, device):

    cfg = SimpleNamespace(**SOH_CONFIG)

    cfg.device = device

    # =====================================================
    # dataset
    # =====================================================
    loader = MITDdataset(cfg)

    if cfg.input_type == "charge":

        data_dict = loader.get_charge_data(
            test_battery_id=cfg.test_battery_id
        )

    elif cfg.input_type == "partial_charge":

        data_dict = loader.get_partial_data(
            test_battery_id=cfg.test_battery_id
        )

    else:

        data_dict = loader.get_features(
            test_battery_id=cfg.test_battery_id
        )

    # =====================================================
    # dataloader
    # =====================================================
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        data_dict
    )



    # =====================================================
    # TS encoder
    # =====================================================
    ts_encoder = LSTMEncoder(
        input_dim=4,
        hidden_dim=128
    ).to(device)

    # =====================================================
    # model
    # =====================================================
    model = TSLLMModel(

        ts_encoder=ts_encoder,

        ts_dim=128,

        llm_path=cfg.llm_path,

        dropout=cfg.dropout,

        device=device,

        llm_dtype="bf16"

    ).to(device)

    print("\n[Model] TSLLMModel")

    # =====================================================
    # optimizer
    # =====================================================
    optimizer = optim.Adam(

        model.parameters(),

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
    # training initialize
    # =====================================================
    best_val = 1e9

    best_metrics = None

    best_epoch = 0

    best_state = None

    train_loss_history = []

    val_loss_history = []

    # =====================================================
    # training start
    # =====================================================
    print("\n========================")
    print("Start Training")
    print("========================")

    for epoch in range(cfg.epochs):

        # =================================================
        # train
        # =================================================
        model.train()

        train_loss_sum = 0.0

        for batch in train_loader:

            # =============================================
            # batch
            # =============================================
            ts_x, y = batch

            ts_x = ts_x.to(device)

            y = y.to(device)

            # =============================================
            # forward
            # =============================================
            pred = model(ts_x)

            pred = pred.view_as(y)

            # =============================================
            # loss
            # =============================================
            loss = loss_fn(pred, y)

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

        train_loss_history.append(train_loss)

        # =================================================
        # validation
        # =================================================
        val_loss, metrics = evaluate(

            model=model,

            device=device,

            loader=val_loader,

            loss_fn=loss_fn
        )

        val_loss_history.append(val_loss)

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
                "model": model.state_dict()
            }

            print(
                f"[Best] "
                f"epoch={best_epoch} "
                f"val_loss={best_val:.6f}"
            )

            print(
                f"        "
                f"MAE={metrics['MAE']:.4f} | "
                f"MAPE={metrics['MAPE']:.2f}% | "
                f"RMSE={metrics['RMSE']:.4f} | "
                f"R2={metrics['R2']:.4f}"
            )

    # =====================================================
    # load best model
    # =====================================================
    print("\n========================")
    print("Load Best Model")
    print("========================")

    model.load_state_dict(
        best_state["model"]
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

    test_loss, metrics = evaluate(

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

    print("\n========================")
    print("Training Finished")
    print("========================")

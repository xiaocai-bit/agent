import torch
from torch import nn, optim
import inspect

from types import SimpleNamespace

from MIT_loader import MITDdataset

from models import (
    LSTMEncoder,
    TSLLMModel,
    PhysicsNet
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

    "input_type": "handcraft_features",

    "batch": 2,

    "batch_size": 32,

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

    "monotonic_lambda": 0.05,

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

    cfg.device = device

    preprocess = getattr(args, "preprocess", None)

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

        try:

            sig = inspect.signature(evaluate)

            if "preprocess" in sig.parameters:
                eval_kwargs["preprocess"] = preprocess

        except (TypeError, ValueError):

            pass

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

    # =====================================================
    # preprocess
    # handcrafted feature + cycle index
    # 67 + 1 = 68
    # =====================================================
    preprocess = nn.Sequential(

        nn.Flatten(),

        nn.Linear(67, 4 * 128),

        nn.GELU(),

        nn.Unflatten(1, (128, 4))

    ).to(device)

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

    # =====================================================
    # physics net
    # =====================================================
    physics_net = PhysicsNet(

        latent_dim=model.llm_enc.hidden_size

    ).to(device)

    print("\n[Model] TSLLMModel + PhysicsNet")

    # =====================================================
    # optimizer
    # =====================================================
    optimizer = optim.AdamW(

        list(model.parameters())
        + list(preprocess.parameters())
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
            ts_x, y, cycle_id = batch

            ts_x = ts_x.float().to(device)

            y = y.float().to(device)

            cycle_id = cycle_id.float().to(device)

            cycle_id.requires_grad_(True)

            # =============================================
            # preprocess
            # =============================================
            ts_x = preprocess(ts_x)

            # =============================================
            # forward
            # =============================================
            pred, latent_h = model(

                ts_x,

                return_latent=True
            )

            pred = pred.unsqueeze(-1)

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

                inputs=cycle_id,

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

                cycle_id
            )

            # =============================================
            # physics loss
            # =============================================
            physics_loss = (

                (dsoh_dn - physics_rhs) ** 2

            ).mean()

            # =============================================
            # monotonic degradation
            # dSOH/dN <= 0
            # =============================================
            monotonic_loss = torch.relu(
                dsoh_dn
            ).mean()

            # =============================================
            # total loss
            # =============================================
            loss = (

                data_loss

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

        # =================================================
        # validation
        # =================================================
        val_loss, metrics = evaluate_with_optional_preprocess(
            val_loader
        )

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

                "preprocess": preprocess.state_dict(),

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

    preprocess.load_state_dict(
        best_state["preprocess"]
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

    test_loss, metrics = evaluate_with_optional_preprocess(
        test_loader
    )

    print(f"test_loss = {test_loss:.6f}")

    print(f"MAE  = {metrics['MAE']:.4f}")

    print(f"MAPE = {metrics['MAPE']:.2f}%")

    print(f"RMSE = {metrics['RMSE']:.4f}")

    print(f"R2   = {metrics['R2']:.4f}")

    print("\n========================")
    print("Training Finished")
    print("========================")

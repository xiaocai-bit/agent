import os
import time
import torch
from torch import nn, optim

from models import LSTMEncoder, TSOnlyModel, SemanticTSModel, AttnThenLLMModel
from train_utils import train_one_epoch, evaluate, ensure_dir, save_best, load_best


def train_soh(args, train_loader, val_loader, test_loader, device):

    # ======================
    # 1. encoder
    # ======================
    ts_encoder = LSTMEncoder(input_dim=4, hidden_dim=128).to(device)

    fusion = args.fusion

    # ======================
    # 2. model
    # ======================
    if not args.use_prompt:
        model = TSOnlyModel(
            ts_encoder=ts_encoder,
            ts_dim=128,
            dropout=args.dropout
        ).to(device)
        mode = "TSonly"

    else:
        if fusion == "film":
            _, _, p0 = next(iter(train_loader))
            prompt_dim = p0.shape[-1]

            model = SemanticTSModel(
                ts_encoder=ts_encoder,
                ts_dim=128,
                prompt_dim=prompt_dim,
                film_hidden=256,
                dropout=args.dropout
            ).to(device)

            mode = "FiLM"

        elif fusion == "crossattn":
            _, _, pt0, _ = next(iter(train_loader))
            prompt_dim = pt0.shape[-1]

            model = AttnThenLLMModel(
                ts_encoder=ts_encoder,
                ts_dim=128,
                prompt_dim=prompt_dim,
                d_model=args.cross_d_model,
                nhead=args.cross_nhead,
                dropout=args.dropout,
                use_llm_feature=args.use_llm_feature,
                llm_path=args.llm_path,
                device=args.device
            ).to(device)

            mode = "CrossAttn"

        else:
            raise ValueError("fusion error")

    # ======================
    # 3. loss / opt
    # ======================
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )

    # ======================
    # 4. save path
    # ======================
    ensure_dir(args.save_dir)

    best_path = os.path.join(args.save_dir, f"best_soh_{mode}.pt")

    best_val = float("inf")
    stop = 0

    # ======================
    # 5. training loop
    # ======================
    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            model, device, train_loader, optimizer,
            loss_fn=loss_fn,
            fusion=fusion,
            use_prompt=args.use_prompt
        )

        scheduler.step()

        val_loss, val_metrics = evaluate(
            model, device, val_loader,
            loss_fn=loss_fn,
            fusion=fusion,
            use_prompt=args.use_prompt
        )

        improved = val_loss < best_val

        print(f"[SOH-{mode}] epoch {epoch} "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"RMSE={val_metrics['RMSE']:.4f}")

        if improved:
            best_val = val_loss
            stop = 0
            save_best(best_path, model, optimizer, epoch, best_val)

            test_loss, test_metrics = evaluate(
                model, device, test_loader,
                loss_fn=loss_fn,
                fusion=fusion,
                use_prompt=args.use_prompt
            )

            print(f"   → TEST RMSE={test_metrics['RMSE']:.4f}")

        else:
            stop += 1

        if stop > args.early_stop:
            print("Early stop")
            break

    # ======================
    # 6. final eval
    # ======================
    load_best(best_path, model, device)

    final_test = evaluate(
        model, device, test_loader,
        loss_fn=loss_fn,
        fusion=fusion,
        use_prompt=args.use_prompt
    )

    print("\n===== FINAL =====")
    print(f"TEST RMSE: {final_test[1]['RMSE']:.4f}")
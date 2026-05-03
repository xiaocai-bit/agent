import torch
import argparse

from task_router import TaskRouter
from SOH import train_soh


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--task_text", type=str, default="")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--use_llm_router", action="store_true")
    parser.add_argument("--qwen_model_path", type=str, default="")

    # training config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)

    # model config
    parser.add_argument("--fusion", default="crossattn")
    parser.add_argument("--use_prompt", action="store_true")

    parser.add_argument("--cross_d_model", type=int, default=256)
    parser.add_argument("--cross_nhead", type=int, default=4)

    parser.add_argument("--early_stop", type=int, default=10)

    args = parser.parse_args()

    # =========================
    # 1. 用户输入（自然语言）
    # =========================
    if not args.task_text:
        print("请输入电池分析任务（自然语言描述）：")
        args.task_text = input(">>> ").strip()

    # =========================
    # 2. Router（任务识别）
    # =========================
    router = TaskRouter()

    task = router.route(
        args.task_text,
        use_llm=args.use_llm_router
    )

    print("\n========================")
    print(f"[Router] → {task.label} ({task.method})")
    print("========================\n")

    # =========================
    # 3. Agent dispatch（核心）
    # =========================
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if task.label == "SOH":
        from MIT_loader import MITDdataset
        from XJTU_loader import XJTUDdataset
        from prompts import build_dataloaders

        def get_data(args):
            if args.data == "MIT":
                loader = MITDdataset(args)
            else:
                loader = XJTUDdataset(args)

            return loader.get_partial_data(test_battery_id=2)

        data_dict = get_data(args)
        train_loader, val_loader, test_loader = build_dataloaders(args, data_dict)

        train_soh(
            args,
            train_loader,
            val_loader,
            test_loader,
            device
        )

    elif task.label == "FD":
        print("FD Agent not implemented yet")

    elif task.label == "AD":
        print("AD Agent not implemented yet")

    else:
        print("Unknown task")


if __name__ == "__main__":
    main()
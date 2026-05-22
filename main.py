
'''import argparse
import torch

from task_router import TaskRouter, LocalQwenClassifier
from SOH import run_soh


def main():

    parser = argparse.ArgumentParser()

    # =========================
    # system-level（必须保留）
    # =========================
    parser.add_argument("--task_text", type=str, default="")
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--llm_router_path",
        type=str,
        default=r"D:\Code_LTF\model_fine_turing\large_model\qwen\Qwen3-1.7B"
    )

    # =========================
    # runtime control（非任务相关）
    # =========================
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--result_root", type=str, default="results")

    args = parser.parse_args()

    # =========================
    # input
    # =========================
    if not args.task_text:
        print("请输入电池分析任务：")
        args.task_text = input(">>> ").strip()

    # =========================
    # router
    # =========================
    llm = LocalQwenClassifier(args.llm_router_path, device=args.device)
    router = TaskRouter(llm=llm)

    task = router.route(args.task_text)

    print("\n========================")
    print(f"[Router] → {task.label}")
    print("========================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # =========================
    # task dispatch
    # =========================
    if task.label == "SOH":
        run_soh(args, device)

    elif task.label in ["FD", "AD"]:
        print(f"⚠️ 已识别任务：{task.label}")
        print("但当前系统暂未实现该模块（仅 SOH 可用）")

        print("\n👉 请尝试：")
        print("- 电池容量为什么下降这么快？")
        print("- 是否存在故障？")
        print("- 是否出现异常波动？")

    else:
        print("⚠️ 未识别为电池任务（UNKNOWN）")

        print("\n👉 请提问电池相关问题，例如：")
        print("- 电池容量为什么下降这么快？")
        print("- 电池是否存在故障？")
        print("- 是否检测到异常波动？")


if __name__ == "__main__":
    main()'''

import argparse
import torch

from task_router import TaskRouter, LocalQwenClassifier
from SOH import run_soh
from AD import run_ad


def main():

    parser = argparse.ArgumentParser()

    # =========================
    # system-level（必须保留）
    # =========================
    parser.add_argument("--task_text", type=str, default="")
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--llm_router_path",
        type=str,
        default=r"D:\Code_LTF\model_fine_turing\large_model\qwen\Qwen3-1.7B"
    )

    # =========================
    # runtime control（非任务相关）
    # =========================
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--result_root", type=str, default="results")

    # AD args
    parser.add_argument("--ad_data_root", type=str, default=r"E:\data\battery-agent\异常监测数据\battery_brand1")
    parser.add_argument("--ad_batch_size", type=int, default=256)
    parser.add_argument("--ad_num_workers", type=int, default=4)
    parser.add_argument("--ad_epochs", type=int, default=20)
    parser.add_argument("--ad_lr", type=float, default=1e-3)
    parser.add_argument("--ad_latent_dim", type=int, default=8)
    parser.add_argument("--ad_n_gmm", type=int, default=4)

    args = parser.parse_args()

    # =========================
    # input
    # =========================
    if not args.task_text:
        print("请输入电池分析任务：")
        args.task_text = input(">>> ").strip()

    # =========================
    # router
    # =========================
    llm = LocalQwenClassifier(args.llm_router_path, device=args.device)
    router = TaskRouter(llm=llm)

    task = router.route(args.task_text)

    print("\n========================")
    print(f"[Router] → {task.label}")
    print("========================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # =========================
    # task dispatch
    # =========================
    if task.label == "SOH":
        run_soh(args, device)

    elif task.label == "AD":
        if not args.ad_data_root:
            print("⚠️ AD 任务需要提供 --ad_data_root")
            return
        run_ad(args, device)

    elif task.label == "FD":
        print("⚠️ 已识别任务：FD（故障诊断）")
        print("但当前系统暂未实现该模块")

    else:
        print("⚠️ 未识别为电池任务（UNKNOWN）")

        print("\n👉 请提问电池相关问题，例如：")
        print("- 电池容量为什么下降这么快？")
        print("- 电池是否存在故障？")
        print("- 是否检测到异常波动？")


if __name__ == "__main__":
    main()

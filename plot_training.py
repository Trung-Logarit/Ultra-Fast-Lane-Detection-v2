"""
Usage:
    python plot_training.py --work_dir /path/to/work_dir

Reads train_steps.csv / train_epochs.csv and regenerates PNG plots.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True, help="Training work_dir containing CSV logs")
    args = ap.parse_args()

    steps_csv = os.path.join(args.work_dir, "train_steps.csv")
    epochs_csv = os.path.join(args.work_dir, "train_epochs.csv")

    if os.path.exists(steps_csv):
        df = pd.read_csv(steps_csv)
        if "loss" in df.columns and "global_step" in df.columns:
            plt.figure()
            plt.plot(df["global_step"], df["loss"])
            plt.xlabel("global_step"); plt.ylabel("loss"); plt.title("Training Loss (step)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.work_dir, "loss_curve_step.png"))
            plt.close()

        if "lr" in df.columns:
            plt.figure()
            plt.plot(df["global_step"], df["lr"])
            plt.xlabel("global_step"); plt.ylabel("learning_rate"); plt.title("LR Schedule")
            plt.tight_layout()
            plt.savefig(os.path.join(args.work_dir, "lr_curve.png"))
            plt.close()

        if "top1" in df.columns and df["top1"].notna().any():
            plt.figure()
            df2 = df.dropna(subset=["top1"])
            plt.plot(df2["global_step"], df2["top1"])
            plt.xlabel("global_step"); plt.ylabel("top1"); plt.title("Top-1 (approx)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.work_dir, "top1_curve.png"))
            plt.close()

            if "top2" in df.columns and df2["top2"].notna().any():
                plt.figure()
                plt.plot(df2["global_step"], df2["top2"])
                plt.xlabel("global_step"); plt.ylabel("top2"); plt.title("Top-2 (approx)")
                plt.tight_layout()
                plt.savefig(os.path.join(args.work_dir, "top2_curve.png"))
                plt.close()

    if os.path.exists(epochs_csv):
        df = pd.read_csv(epochs_csv)
        if "avg_loss" in df.columns and "epoch" in df.columns:
            plt.figure()
            plt.plot(df["epoch"], df["avg_loss"])
            plt.xlabel("epoch"); plt.ylabel("avg_loss"); plt.title("Training Loss (epoch)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.work_dir, "loss_curve_epoch.png"))
            plt.close()

    print("Plots saved to:", args.work_dir)

if __name__ == "__main__":
    main()

"""Resume GenSHAP-S SVM prediction diagnostics from a saved checkpoint.

This script intentionally does not train. It imports the model definition and
preprocessing helpers from GenSHAP_S_SVM.py, loads the saved checkpoint, and
generates missing diagnostics under Outputs_II/260522/Trial00.
"""

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.exceptions import ConvergenceWarning


DEFAULT_HOLDOUTS = [
    "Iris_1",
    "Vehicle_1",
    "STL10_27xxx_2",
    "STL10_1279x_1",
    "volkert_25_1",
    "volkert_0678_2",
    "BC_2",
]


def load_training_module():
    """Import GenSHAP_S_SVM.py without triggering its __main__ training block."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import torch

    original_set_device = None
    if not torch.cuda.is_available():
        original_set_device = torch.cuda.set_device
        torch.cuda.set_device = lambda *_args, **_kwargs: None

    try:
        import GenSHAP_S_SVM as train_mod
    finally:
        if original_set_device is not None:
            torch.cuda.set_device = original_set_device

    return train_mod


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_complete_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def remove_if_exists(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def to_csv_atomic(df: pd.DataFrame, path: str, **kwargs):
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    try:
        df.to_csv(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    finally:
        remove_if_exists(tmp_path)


def savefig_atomic(path: str, *, dpi: int, fig=None):
    ensure_dir(os.path.dirname(path))
    root, ext = os.path.splitext(path)
    ext = ext or ".png"
    tmp_path = f"{root}.tmp{ext}"
    try:
        if fig is None:
            plt.savefig(tmp_path, dpi=dpi)
        else:
            fig.savefig(tmp_path, dpi=dpi)
        os.replace(tmp_path, path)
    finally:
        remove_if_exists(tmp_path)


def load_gnn_from_checkpoint(train_mod, model_path: str):
    torch = train_mod.torch
    ckpt = torch.load(model_path, map_location=train_mod.device)
    in_channels = ckpt.get("in_channels", train_mod.svd_d)
    hidden_channels = ckpt.get("hidden_channels", 64)
    heads = ckpt.get("heads", 2)

    gnn = train_mod.GINShap(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
    )
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    gnn.load_state_dict(state)
    gnn.to(train_mod.device)
    gnn.eval()
    return gnn


def performance_plot_removal(train_mod, vals, X_train, y_train, X_test, y_test,
                             a, b, *, order="descending", random_iters=10,
                             ylim=None, save_path=None, random_seed=23):
    if isinstance(vals, dict):
        labels, val_arrays = list(vals.keys()), list(vals.values())
    else:
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        val_arrays = list(vals)
        labels = [getattr(v, "name", None) for v in val_arrays]
        labels = [lbl if lbl else f"valuation {i + 1}" for i, lbl in enumerate(labels)]

    n = len(X_train)
    total_remove = int(np.floor(b * n))
    step_remove = max(1, int(np.floor(a * n)))
    removal_counts = list(range(0, total_remove + 1, step_remove))
    removal_percentages = [100 * r / n for r in removal_counts]

    def curve_for_ordering(ordering):
        curve = []
        for r in removal_counts:
            keep = ordering[r:]
            if len(keep) == 0:
                curve.append(0.0)
                continue
            X_sub, y_sub = X_train[keep], y_train[keep]
            if len(np.unique(y_sub)) < 2:
                curve.append(float(np.mean(np.full_like(y_test, y_sub[0]) == y_test)))
                continue
            model = train_mod.build_svm_model(max_iter=200)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                model.fit(X_sub, y_sub)
            curve.append(float(model.score(X_test, y_test)))
        return np.asarray(curve)

    results = {}
    for label, val in zip(labels, val_arrays):
        order_idx = np.argsort(val)
        if order != "ascending":
            order_idx = order_idx[::-1]
        results[label] = curve_for_ordering(order_idx)

    rng = np.random.default_rng(random_seed)
    random_curves = [curve_for_ordering(rng.permutation(n)) for _ in range(random_iters)]
    results["Random"] = np.mean(random_curves, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150, layout="constrained")
    ax.set_xlabel("Fraction of data removed (%)")
    ax.set_ylabel("Accuracy (%)")
    for key, curve in results.items():
        ax.plot(removal_percentages, curve * 100, lw=5, marker="o", label=key)
    ax.grid(True)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if save_path is not None:
        savefig_atomic(save_path, dpi=150, fig=fig)
    plt.close(fig)
    return removal_percentages, results


def diagnose_train_sets(train_mod, gnn, base_train_dir: str, out_dir: str,
                        *, resume: bool, continue_on_error: bool):
    ensure_dir(out_dir)
    triplets = train_mod.discover_datasets(base_train_dir, require_tmc=False)

    for item in triplets:
        train_csv, test_csv = item["train_like"], item["test"]
        base_name = os.path.basename(train_csv).replace(".csv", "")
        out_csv = os.path.join(out_dir, f"{base_name}_Pred.csv")
        out_png = os.path.join(out_dir, f"{base_name}_ShapleyCompare.png")

        if resume and is_complete_file(out_csv) and is_complete_file(out_png):
            print("Skipping complete train diagnostics for:", base_name, flush=True)
            continue

        try:
            df_train = None
            if resume and is_complete_file(out_csv):
                try:
                    df_train = pd.read_csv(out_csv)
                    if "Pred_Shap" not in df_train.columns:
                        df_train = None
                    else:
                        print("Reusing:", out_csv, flush=True)
                except Exception:
                    print("WARNING: Existing prediction CSV is unreadable; regenerating:", out_csv, flush=True)
                    df_train = None

            if df_train is None:
                dlist = train_mod.load_and_preprocess(train_csv, test_csv, d=train_mod.svd_d)
                preds = train_mod.predict(gnn, dlist)
                df_train = pd.read_csv(train_csv)
                df_train["Pred_Shap"] = preds
                to_csv_atomic(df_train, out_csv, index=False)
                print("Saved to:", out_csv, flush=True)

            if resume and is_complete_file(out_png):
                print("Skipping existing plot:", out_png, flush=True)
                continue

            true_vals = df_train["PerDShap"].to_numpy()
            pred_vals = df_train["Pred_Shap"].to_numpy()
            y_labels = (
                df_train["target"].to_numpy().astype(int)
                if "target" in df_train.columns
                else np.zeros_like(true_vals)
            )
            mean_true, mean_pred = true_vals.mean(), pred_vals.mean()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            bins = 50
            ax1.hist(true_vals, bins=bins, alpha=0.55,
                     label="Permutated-based Data Shapley",
                     edgecolor="k", linewidth=0.3, color=train_mod.COLORS["true"])
            ax1.hist(pred_vals, bins=bins, alpha=0.55,
                     label="Predicted Data Shapley",
                     edgecolor="k", linewidth=0.3, color=train_mod.COLORS["pred"])
            ax1.axvline(mean_true, linestyle="--", linewidth=2,
                        color=train_mod.COLORS["true"], label=f"Mean True: {mean_true:.3f}")
            ax1.axvline(mean_pred, linestyle="--", linewidth=2,
                        color=train_mod.COLORS["pred"], label=f"Mean Pred: {mean_pred:.3f}")
            ax1.set_xlabel("Data Shapley Value", fontsize=12, weight="bold")
            ax1.set_ylabel("Count", fontsize=12, weight="bold")
            ax1.set_title("Permutated-based vs Predicted Data Shapley (Histogram)",
                          fontsize=13, weight="bold")
            ax1.grid(axis="y", linestyle="--", alpha=0.4)
            ax1.legend(fontsize=10)

            lo = min(true_vals.min(), pred_vals.min())
            hi = max(true_vals.max(), pred_vals.max())
            pad = 0.005 * (hi - lo if hi > lo else 1.0)
            ax2.set_xlim(lo - pad, hi + pad)
            ax2.set_ylim(lo - pad, hi + pad)
            ax2.set_aspect("equal", adjustable="box")

            classes = np.unique(y_labels)
            styles = train_mod._class_styles(classes)
            for cls in classes:
                mask = y_labels == cls
                style = styles[cls].copy()
                ax2.scatter(true_vals[mask], pred_vals[mask], **style,
                            facecolors="none", edgecolors=train_mod.COLORS["pred"],
                            linewidths=1.2)

            ax2.plot([lo, hi], [lo, hi], "--k", linewidth=1.5)
            ax2.set_xlabel("Permutated-based Data Shapley Values", fontsize=12, weight="bold")
            ax2.set_ylabel("Predicted Data Shapley Values", fontsize=12, weight="bold")
            ax2.set_title("True vs Predicted (Scatter)", fontsize=13, weight="bold")
            ax2.grid(True, linestyle="--", alpha=0.4)

            class_handles = [
                Line2D([0], [0], marker=styles[cls]["marker"], linestyle="None",
                       markerfacecolor="none", markeredgecolor="k",
                       markersize=8, label=f"class {cls}")
                for cls in classes
            ]
            ax2.legend(handles=class_handles, title="Class (marker)", fontsize=9, loc="best")

            fig.suptitle(f"Shapley Comparison: {os.path.basename(train_csv)}",
                         fontsize=15, weight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            savefig_atomic(out_png, dpi=300, fig=fig)
            plt.close(fig)
            print("Saved to:", out_png, flush=True)
        except Exception:
            plt.close("all")
            print(f"ERROR: train diagnostics failed for {base_name}", flush=True)
            traceback.print_exc()
            if not continue_on_error:
                raise


def diagnose_test_sets(train_mod, gnn, base_test_dir: str, holdout_bases, out_dir: str,
                       *, a: float, b: float, ylim, random_iters: int,
                       resume: bool, continue_on_error: bool):
    ensure_dir(out_dir)
    catalog = {
        item["base_name"]: item
        for item in train_mod.discover_datasets(base_test_dir, require_tmc=False)
    }

    for base in holdout_bases:
        if base not in catalog:
            print(f"WARNING: Base {base} not found in {base_test_dir}.", flush=True)
            continue

        paths = catalog[base]
        out_pred_csv = os.path.join(out_dir, f"{base}_Pred.csv")
        desc_png = os.path.join(out_dir, f"{base}_Removal_High.png")
        asc_png = os.path.join(out_dir, f"{base}_Removal_Low.png")
        hist_png = os.path.join(out_dir, f"{base}_DShapHistogram.png")
        line_png = os.path.join(out_dir, f"{base}_LinearCompare.png")

        try:
            df_tr_like = None
            if resume and is_complete_file(out_pred_csv):
                try:
                    df_tr_like = pd.read_csv(out_pred_csv)
                    if "Pred_Shap" not in df_tr_like.columns:
                        df_tr_like = None
                    else:
                        print("Reusing:", out_pred_csv, flush=True)
                except Exception:
                    print("WARNING: Existing prediction CSV is unreadable; regenerating:",
                          out_pred_csv, flush=True)
                    df_tr_like = None

            if df_tr_like is None:
                dlist = train_mod.load_and_preprocess(paths["train_like"], paths["test"],
                                                      d=train_mod.svd_d)
                preds = train_mod.predict(gnn, dlist)
                df_tr_like = pd.read_csv(paths["train_like"])
                df_tr_like["Pred_Shap"] = preds
                to_csv_atomic(df_tr_like, out_pred_csv, index=False)
                print("Saved to:", out_pred_csv, flush=True)

            if paths.get("tmc") is None:
                print(f"WARNING: No TMC file for {base}.", flush=True)
                continue

            if (resume and is_complete_file(desc_png) and is_complete_file(asc_png)
                    and is_complete_file(hist_png) and is_complete_file(line_png)):
                print("Skipping complete test diagnostics for:", base, flush=True)
                continue

            df_test = pd.read_csv(paths["test"])
            X_train = df_tr_like.drop(columns=["target", "PerDShap", "Pred_Shap"]).astype(float)
            X_test = df_test.drop(columns=["target"]).astype(float)
            y_train = df_tr_like["target"].to_numpy().astype(int)
            y_test = df_test["target"].to_numpy().astype(int)

            df_tmc = pd.read_csv(paths["tmc"])
            tmc_vals = df_tmc["TMC_Shap"].to_numpy()
            per_vals = df_tr_like["PerDShap"].to_numpy()
            pred_vals = df_tr_like["Pred_Shap"].to_numpy()

            plot_vals = {
                "Permuted Data Shapley": per_vals,
                "TMC Data Shapley": tmc_vals,
                "Predicted Shapley": pred_vals,
            }

            if not (resume and is_complete_file(desc_png)):
                performance_plot_removal(
                    train_mod, vals=plot_vals,
                    X_train=X_train.to_numpy(), y_train=y_train,
                    X_test=X_test.to_numpy(), y_test=y_test,
                    a=a, b=b, ylim=ylim, order="descending",
                    random_iters=random_iters, save_path=desc_png,
                )
                print("Saved to:", desc_png, flush=True)

            if not (resume and is_complete_file(asc_png)):
                performance_plot_removal(
                    train_mod, vals=plot_vals,
                    X_train=X_train.to_numpy(), y_train=y_train,
                    X_test=X_test.to_numpy(), y_test=y_test,
                    a=a, b=b, ylim=ylim, order="ascending",
                    random_iters=random_iters, save_path=asc_png,
                )
                print("Saved to:", asc_png, flush=True)

            if not (resume and is_complete_file(hist_png)):
                mean_tmc, mean_per, mean_pred = tmc_vals.mean(), per_vals.mean(), pred_vals.mean()
                fig = plt.figure(figsize=(11, 8))
                bins = 50
                plt.hist(per_vals, bins=bins, alpha=0.45,
                         label="Permutated-based Data Shapley",
                         edgecolor="k", linewidth=0.3, color=train_mod.COLORS["true"])
                plt.hist(tmc_vals, bins=bins, alpha=0.45, label="TMC Data Shapley",
                         edgecolor="k", linewidth=0.3, color=train_mod.COLORS["tmc"])
                plt.hist(pred_vals, bins=bins, alpha=0.45, label="Predicted Data Shapley",
                         edgecolor="k", linewidth=0.3, color=train_mod.COLORS["pred"])
                plt.axvline(mean_per, linestyle="--", linewidth=2,
                            color=train_mod.COLORS["true"], label=f"Avg Permuted ({mean_per:.3f})")
                plt.axvline(mean_tmc, linestyle="--", linewidth=2,
                            color=train_mod.COLORS["tmc"], label=f"Avg TMC ({mean_tmc:.3f})")
                plt.axvline(mean_pred, linestyle="--", linewidth=2,
                            color=train_mod.COLORS["pred"], label=f"Avg Pred ({mean_pred:.3f})")
                plt.xlabel("Shapley Value", fontsize=10)
                plt.ylabel("Count", fontsize=10)
                plt.title(f"Distribution of Shapley Values - {base}", fontsize=11)
                plt.xticks(fontsize=9)
                plt.yticks(fontsize=9)
                plt.grid(axis="y", linestyle="--", alpha=0.4)
                plt.legend(fontsize=8)
                plt.tight_layout()
                savefig_atomic(hist_png, dpi=500, fig=fig)
                plt.close(fig)
                print("Saved to:", hist_png, flush=True)

            if not (resume and is_complete_file(line_png)):
                lo = min(per_vals.min(), tmc_vals.min(), pred_vals.min())
                hi = max(per_vals.max(), tmc_vals.max(), pred_vals.max())
                classes = np.unique(y_train)
                styles = train_mod._class_styles(classes)

                fig, ax = plt.subplots(figsize=(8.8, 8.8))
                for cls in classes:
                    mask = y_train == cls
                    style = styles[cls].copy()
                    ax.scatter(per_vals[mask], pred_vals[mask], **style,
                               facecolors="none", edgecolors=train_mod.COLORS["pred"],
                               linewidths=1.2)
                    ax.scatter(per_vals[mask], tmc_vals[mask], **style,
                               facecolors="none", edgecolors=train_mod.COLORS["tmc"],
                               linewidths=1.2)

                ax.plot([lo - 0.001, hi + 0.001], [lo - 0.001, hi + 0.001],
                        "--k", linewidth=1.5)
                ax.set_xlabel("Permuted Shapley Value")
                ax.set_ylabel("Other Shapley Values")
                ax.set_title(f"Linear Comparison - {base}")
                ax.grid(True, linestyle="--", alpha=0.5)

                series_handles = [
                    Line2D([0], [0], marker="o", linestyle="None",
                           markerfacecolor="none",
                           markeredgecolor=train_mod.COLORS["pred"],
                           markersize=8, label="Predicted Shapley"),
                    Line2D([0], [0], marker="o", linestyle="None",
                           markerfacecolor="none",
                           markeredgecolor=train_mod.COLORS["tmc"],
                           markersize=8, label="TMC Shapley"),
                ]
                legend_series = ax.legend(handles=series_handles, title="Series (color)",
                                          loc="upper left", fontsize=9)
                ax.add_artist(legend_series)

                class_handles = [
                    Line2D([0], [0], marker=styles[cls]["marker"], linestyle="None",
                           markerfacecolor="none", markeredgecolor="k",
                           markersize=8, label=f"class {cls}")
                    for cls in classes
                ]
                legend_classes = ax.legend(handles=class_handles, title="Class (marker)",
                                           loc="lower right", fontsize=9)
                ax.add_artist(legend_classes)

                plt.tight_layout()
                savefig_atomic(line_png, dpi=500, fig=fig)
                plt.close(fig)
                print("Saved to:", line_png, flush=True)
        except Exception:
            plt.close("all")
            print(f"ERROR: test diagnostics failed for {base}", flush=True)
            traceback.print_exc()
            if not continue_on_error:
                raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", default=None)
    parser.add_argument("--base-test", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--holdouts", default=",".join(DEFAULT_HOLDOUTS),
                        help="Comma-separated holdout base names.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Regenerate outputs even when files already exist.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop at the first dataset error.")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--a", type=float, default=0.05)
    parser.add_argument("--b", type=float, default=0.40)
    parser.add_argument("--ylim", type=float, nargs=2, default=(40, 105))
    parser.add_argument("--random-iters", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    train_mod = load_training_module()

    base_train = args.base_train or str(train_mod.DATASETS_DIR / "Base_SVM" / "Train")
    base_test = args.base_test or str(train_mod.DATASETS_DIR / "Base_SVM" / "Test")
    out_dir = args.out_dir or str(train_mod.OUTPUTS_DIR / "260522" / "Trial00")
    model_path = args.model_path or os.path.join(out_dir, "GenSHAP_S_SVM_model.pt")
    holdouts = [item.strip() for item in args.holdouts.split(",") if item.strip()]

    print("Loading model:", model_path, flush=True)
    gnn = load_gnn_from_checkpoint(train_mod, model_path)
    resume = not args.no_resume
    continue_on_error = not args.stop_on_error

    if not args.test_only:
        train_dir = os.path.join(out_dir, "diagnostics_train_final")
        diagnose_train_sets(
            train_mod, gnn, base_train, train_dir,
            resume=resume, continue_on_error=continue_on_error,
        )

    if not args.train_only:
        test_dir = os.path.join(out_dir, "diagnostics_test_final")
        diagnose_test_sets(
            train_mod, gnn, base_test, holdouts, test_dir,
            a=args.a, b=args.b, ylim=tuple(args.ylim),
            random_iters=args.random_iters,
            resume=resume, continue_on_error=continue_on_error,
        )


if __name__ == "__main__":
    main()

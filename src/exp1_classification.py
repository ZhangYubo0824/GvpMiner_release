#!/usr/bin/env python3
"""
实验1: 双方法分类对比 (Supplementary Benchmark)
================================================
使用两种蛋白语言模型表征方法 + 统一 Random Forest 分类器对 Gvp 家族进行多类别分类。
作为方法可靠性的验证基准，结果放入论文 Supplementary。

用法:
    python exp1_classification.py                    # 运行全部方法
    python exp1_classification.py --methods M2       # 仅运行ESM-2
    python exp1_classification.py --methods M2 M3    # 运行ESM-2和ProtBERT

输出:
    work/results/exp1_classification/
        exp1_results_summary.csv       — 方法性能对比表
        exp1_confusion_matrix_*.png    — 混淆矩阵
        exp1_classification_comparison.png — 柱状图对比
        embeddings_M*.npy              — 缓存的嵌入矩阵 (复用)
"""

import sys
import csv
import json
import pickle
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import (
    EXP1_DIR, PREPROCESSED_DIR,
    EXP1_CV_FOLDS, EXP1_RF_ESTIMATORS, EXP1_RANDOM_STATE,
    MIN_FAMILY_SIZE,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


def load_dataset():
    """加载预处理后的数据集"""
    csv_path = PREPROCESSED_DIR / "cleaned_gvp_dataset.csv"
    if not csv_path.exists():
        print(f"错误: 找不到 {csv_path}")
        print("请先运行 data_preprocessing.py")
        sys.exit(1)

    sequences = []
    labels = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("sequence", "").strip()
            family = row.get("family", "").strip()
            if seq and family:
                sequences.append(seq)
                labels.append(family)

    print(f"加载数据: {len(sequences)} 条序列, {len(set(labels))} 个家族")

    # 过滤样本过少的家族
    family_counts = Counter(labels)
    valid_families = {f for f, c in family_counts.items() if c >= MIN_FAMILY_SIZE}
    mask = [l in valid_families for l in labels]
    sequences = [s for s, m in zip(sequences, mask) if m]
    labels = [l for l, m in zip(labels, mask) if m]

    removed_families = set(family_counts.keys()) - valid_families
    if removed_families:
        print(f"过滤掉样本不足的家族: {', '.join(sorted(removed_families))}")

    print(f"有效数据: {len(sequences)} 条, {len(set(labels))} 个家族")
    return sequences, labels


def get_embedder(method_name):
    """根据方法名创建嵌入器"""
    if method_name == "M2":
        from embeddings.esm2 import ESM2Embedder
        return ESM2Embedder()
    elif method_name == "M3":
        from embeddings.protbert import ProtBERTEmbedder
        return ProtBERTEmbedder()
    else:
        raise ValueError(f"未知方法: {method_name}")


def extract_or_load_embeddings(method_name, sequences):
    """提取或从缓存加载嵌入"""
    cache_path = EXP1_DIR / f"embeddings_{method_name}.npy"

    if cache_path.exists():
        print(f"  从缓存加载: {cache_path}")
        return np.load(cache_path)

    print(f"  提取 {method_name} 嵌入...")
    embedder = get_embedder(method_name)

    t0 = time.time()
    X = embedder.embed_batch(sequences)
    elapsed = time.time() - t0

    print(f"  完成: {X.shape}, 耗时 {elapsed:.1f}s")

    # 缓存
    np.save(cache_path, X)
    print(f"  已缓存: {cache_path}")

    return X


def evaluate_classification(X, y_encoded, label_encoder, method_name):
    """5-Fold CV 评估分类性能"""
    skf = StratifiedKFold(n_splits=EXP1_CV_FOLDS, shuffle=True,
                           random_state=EXP1_RANDOM_STATE)

    all_y_true = []
    all_y_pred = []
    prediction_records = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        clf = RandomForestClassifier(
            n_estimators=EXP1_RF_ESTIMATORS,
            random_state=EXP1_RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y_encoded[train_idx])
        y_pred = clf.predict(X[test_idx])

        acc = accuracy_score(y_encoded[test_idx], y_pred)
        f1 = f1_score(y_encoded[test_idx], y_pred, average="macro")

        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "f1_macro": f1})
        all_y_true.extend(y_encoded[test_idx])
        all_y_pred.extend(y_pred)
        true_labels = label_encoder.inverse_transform(y_encoded[test_idx])
        pred_labels = label_encoder.inverse_transform(y_pred)
        for sample_index, true_code, pred_code, true_label, pred_label in zip(
            test_idx,
            y_encoded[test_idx],
            y_pred,
            true_labels,
            pred_labels,
        ):
            prediction_records.append(
                {
                    "method": method_name,
                    "fold": fold + 1,
                    "sample_index": int(sample_index),
                    "true_label": str(true_label),
                    "predicted_label": str(pred_label),
                    "true_code": int(true_code),
                    "predicted_code": int(pred_code),
                    "correct": bool(true_code == pred_code),
                }
            )

        print(f"    Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}")

    # 汇总
    mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
    std_acc = np.std([m["accuracy"] for m in fold_metrics])
    mean_f1 = np.mean([m["f1_macro"] for m in fold_metrics])
    std_f1 = np.std([m["f1_macro"] for m in fold_metrics])

    # 混淆矩阵 (全部fold汇总)
    cm = confusion_matrix(all_y_true, all_y_pred)

    # 分类报告
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    results = {
        "method": method_name,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "fold_metrics": fold_metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "prediction_records": prediction_records,
    }

    print(f"  {method_name}: Acc={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}±{std_f1:.4f}")
    return results


def get_classifier_bundle_path(method_name):
    import sklearn

    skver = sklearn.__version__.replace(".", "_")
    return EXP1_DIR / f"classifier_bundle_{method_name}_sk{skver}.pkl"


def save_final_classifier_bundle(X, y_encoded, label_encoder, method_name):
    """Train and save the final classifier bundle for downstream reuse."""
    EXP1_DIR.mkdir(parents=True, exist_ok=True)
    clf = RandomForestClassifier(
        n_estimators=EXP1_RF_ESTIMATORS,
        random_state=EXP1_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X, y_encoded)

    bundle_path = get_classifier_bundle_path(method_name)
    with open(bundle_path, "wb") as f:
        pickle.dump({"classifier": clf, "label_encoder": label_encoder, "method": method_name}, f)

    print(f"  已保存分类器: {bundle_path}")
    return bundle_path

def _write_matrix_csv(path, matrix, class_names):
    """Write a square confusion matrix with explicit true-label row names."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row])


def _row_normalize_matrix(matrix):
    """Return row-normalized matrix values, preserving all-zero rows."""
    values = np.asarray(matrix, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        row_sums,
        out=np.zeros_like(values, dtype=float),
        where=row_sums != 0,
    )


def save_confusion_matrix_source_tables(all_results, label_encoder):
    """Save raw and row-normalized confusion matrices for future plotting."""
    class_names = list(label_encoder.classes_)
    long_rows = []
    for result in all_results:
        method = result["method"]
        matrix = np.asarray(result["confusion_matrix"], dtype=int)
        normalized = _row_normalize_matrix(matrix)
        _write_matrix_csv(
            EXP1_DIR / f"confusion_matrix_{method}.csv",
            matrix,
            class_names,
        )
        _write_matrix_csv(
            EXP1_DIR / f"confusion_matrix_{method}_row_normalized.csv",
            normalized,
            class_names,
        )
        np.save(EXP1_DIR / f"confusion_matrix_{method}.npy", matrix)
        for true_label, raw_row, norm_row in zip(class_names, matrix, normalized):
            for predicted_label, count, fraction in zip(class_names, raw_row, norm_row):
                long_rows.append(
                    {
                        "method": method,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(count),
                        "row_fraction": float(fraction),
                    }
                )

    long_path = EXP1_DIR / "exp1_confusion_matrix_long.csv"
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "true_label",
                "predicted_label",
                "count",
                "row_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"混淆矩阵数据: {long_path}")


def save_prediction_records(all_results):
    """Save per-sample cross-validation predictions when available."""
    rows = [
        record
        for result in all_results
        for record in result.get("prediction_records", [])
    ]
    if not rows:
        return
    path = EXP1_DIR / "exp1_cross_validation_predictions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "fold",
                "sample_index",
                "true_label",
                "predicted_label",
                "true_code",
                "predicted_code",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"交叉验证预测数据: {path}")


def save_results(all_results, label_encoder):
    """保存所有结果"""
    EXP1_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 汇总表
    summary_path = EXP1_DIR / "exp1_results_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "mean_acc", "std_acc", "mean_f1", "std_f1"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "method": r["method"],
                "mean_acc": f"{r['mean_acc']:.4f}",
                "std_acc": f"{r['std_acc']:.4f}",
                "mean_f1": f"{r['mean_f1']:.4f}",
                "std_f1": f"{r['std_f1']:.4f}",
            })
    print(f"\n汇总表: {summary_path}")

    # 2. 混淆矩阵图
    from utils.visualization import plot_confusion_matrices
    cms = [r["confusion_matrix"] for r in all_results]
    method_names = [r["method"] for r in all_results]
    plot_confusion_matrices(
        cms, method_names, label_encoder.classes_,
        EXP1_DIR / "exp1_confusion_matrices.png",
    )
    save_confusion_matrix_source_tables(all_results, label_encoder)
    save_prediction_records(all_results)

    # 3. 性能对比图
    from utils.visualization import plot_classification_comparison
    comparison_dict = {
        r["method"]: {"acc": r["mean_acc"], "f1": r["mean_f1"]}
        for r in all_results
    }
    plot_classification_comparison(
        comparison_dict,
        EXP1_DIR / "exp1_classification_comparison.png",
    )

    # 4. 详细分类报告 (JSON)
    report_path = EXP1_DIR / "exp1_detailed_reports.json"
    reports = {}
    for r in all_results:
        reports[r["method"]] = {
            "accuracy": r["mean_acc"],
            "f1_macro": r["mean_f1"],
            "per_class": {
                cls: {
                    "precision": r["classification_report"].get(cls, {}).get("precision", 0),
                    "recall": r["classification_report"].get(cls, {}).get("recall", 0),
                    "f1-score": r["classification_report"].get(cls, {}).get("f1-score", 0),
                }
                for cls in label_encoder.classes_
            }
        }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"详细报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="实验1: 双方法分类对比")
    parser.add_argument("--methods", nargs="+", default=["M2", "M3"],
                        choices=["M2", "M3"],
                        help="选择运行的方法 (默认全部)")
    args = parser.parse_args()

    print("=" * 60)
    print("实验1: 两种方法 Gvp 家族分类对比")
    print("=" * 60)

    # 加载数据
    sequences, labels = load_dataset()

    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"类别: {list(le.classes_)}")

    # 对每种方法提取嵌入并评估
    all_results = []

    for method in args.methods:
        print(f"\n{'─'*50}")
        print(f"方法: {method}")
        print(f"{'─'*50}")

        X = extract_or_load_embeddings(method, sequences)

        # 检查嵌入是否有效
        if np.isnan(X).any():
            print(f"  [警告] {method} 嵌入含NaN, 替换为0")
            X = np.nan_to_num(X)

        # 降维可视化
        from utils.visualization import plot_dimensionality_reduction
        dim_reduction_path = EXP1_DIR / f"exp1_dim_reduction_{method}.png"
        plot_dimensionality_reduction(X, labels, method, le.classes_, dim_reduction_path)

        result = evaluate_classification(X, y, le, method)
        all_results.append(result)
        save_final_classifier_bundle(X, y, le, method)

    # 保存结果
    print(f"\n{'='*60}")
    print("保存结果")
    print(f"{'='*60}")
    save_results(all_results, le)

    # 打印最终对比
    print(f"\n{'='*60}")
    print("最终对比")
    print(f"{'='*60}")
    print(f"{'方法':<15} {'Accuracy':<20} {'Macro F1':<20}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['method']:<15} "
              f"{r['mean_acc']:.4f}±{r['std_acc']:.4f}      "
              f"{r['mean_f1']:.4f}±{r['std_f1']:.4f}")

    print("\n实验1完成! ✓")


if __name__ == "__main__":
    main()

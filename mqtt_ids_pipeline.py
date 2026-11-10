"""
Prototype ML pipeline for an MQTT intrusion detection system.
Dataset: MQTT-IoT-IDS2020 (packet / uniflow / biflow CSV features).

Usage:
    python mqtt_ids_pipeline.py --data-dir data/biflow_features --level biflow
    python mqtt_ids_pipeline.py --data-dir data/packet_features --level packet --sample 0.05 --max-per-class 100000
Each CSV in data-dir is one scenario (normal, scan_A, scan_sU, sparta, mqtt_bruteforce);
the is_attack column gives the label of every row.
"""
import argparse, glob, os, random, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Identifier columns are removed so the model learns behaviour, not addresses
ID_COLUMNS = ["ip_src", "ip_dst", "prt_src", "prt_dst", "src_ip", "dst_ip", "src_port", "dst_port",
              "tcp_srcport", "tcp_dstport", "udp_srcport", "udp_dstport", "timestamp", "frame_time"]
KNN_MAX_TRAIN = 200_000   # k-NN prediction is very slow on bigger data


def load_dataset(data_dir, sample_frac=None, max_per_class=None, seed=42):
    frames = []
    rnd = random.Random(seed)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        # random sample of rows (not the first N rows: files are ordered in time)
        skip = (lambda i: i > 0 and rnd.random() > sample_frac) if sample_frac else None
        df = pd.read_csv(path, low_memory=False, skiprows=skip)
        # some CSVs contain repeated header lines inside the data (files were concatenated)
        header_rows = (df.astype(str) == pd.Series(df.columns, index=df.columns)).all(axis=1)
        if header_rows.any():
            print(f"  {os.path.basename(path)}: removed {int(header_rows.sum())} repeated header rows")
            df = df[~header_rows]
        # columns read as text because of those rows -> back to numbers
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() >= 0.95 * df[c].notna().sum():
                    df[c] = conv
        if "is_attack" in df.columns:
            df = df[df["is_attack"].notna()]
        # e.g. "biflow_mqtt_bruteforce" -> "mqtt_bruteforce"
        name = os.path.splitext(os.path.basename(path))[0].lower()
        for prefix in ("biflow_", "uniflow_", "packet_"):
            name = name.replace(prefix, "")
        if "is_attack" in df.columns:
            # attack files also contain normal background traffic,
            # so the dataset's own is_attack column is the ground truth
            df["label"] = df["is_attack"].astype(int)
            df = df.drop(columns=["is_attack"])               # never use the answer as a feature
        else:
            df["label"] = 0 if "normal" in name else 1
        df["attack_type"] = np.where(df["label"] == 1, name, "normal")  # for multiclass runs
        frames.append(df)
        print(f"{name:18s} rows={len(df):>9,}  attacks={int(df['label'].sum()):>9,}")
    data = pd.concat(frames, ignore_index=True)
    if max_per_class:   # balance classes: at most N rows of each traffic type
        parts = [g.sample(min(len(g), max_per_class), random_state=seed)
                 for _, g in data.groupby("attack_type")]
        data = pd.concat(parts, ignore_index=True)
        print("Rows per class after balancing:", data["attack_type"].value_counts().to_dict())
    data = data.drop(columns=[c for c in ID_COLUMNS if c in data.columns])
    data = data.dropna(axis=1, how="all")
    # text columns: few values (e.g. proto TCP/UDP) -> one-hot, otherwise dropped
    text_cols = [c for c in data.columns
                 if not pd.api.types.is_numeric_dtype(data[c]) and c not in ("attack_type", "label")]
    onehot = [c for c in text_cols if data[c].nunique() <= 20]
    dropped = [c for c in text_cols if c not in onehot]
    data = pd.get_dummies(data.drop(columns=dropped), columns=onehot, dtype=int)
    # empty values mean "field not present" (e.g. MQTT fields in a non-MQTT packet) -> 0
    data = data.replace([np.inf, -np.inf], np.nan)
    num_cols = data.columns.drop(["label", "attack_type"])
    data[num_cols] = data[num_cols].fillna(0)
    if dropped:
        print("Dropped text columns:", dropped)
    return data


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate(data, multiclass=False, seed=42):
    X = data.drop(columns=["label", "attack_type"]).select_dtypes(include=[np.number])
    y = data["attack_type"] if multiclass else data["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
    }
    rows = []
    for name, model in models.items():
        if name == "kNN" and len(X_tr) > KNN_MAX_TRAIN:
            print(f"kNN skipped: {len(X_tr):,} training rows > {KNN_MAX_TRAIN:,} (too slow)")
            continue
        print(f"Training {name}...", flush=True)
        model.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        y_pred = model.predict(X_te)
        ms_per_sample = (time.perf_counter() - t0) * 1000 / len(X_te)
        avg = "macro" if multiclass else "binary"
        p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average=avg, zero_division=0)
        if multiclass:   # FPR = normal traffic labelled as any attack
            fpr = false_positive_rate((y_te != "normal").astype(int), (y_pred != "normal").astype(int))
        else:
            fpr = false_positive_rate(y_te, y_pred)
        rows.append({"model": name, "precision": p, "recall": r, "f1": f1,
                     "fpr": fpr, "inference_ms": ms_per_sample})
        if name == "RandomForest":   # top features: helps to spot data leakage
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("Top-10 features (Random Forest):")
            print(imp.head(10).round(4).to_string(), "\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--level", default="biflow", choices=["packet", "uniflow", "biflow"])
    ap.add_argument("--sample", type=float, default=None,
                    help="random share of rows to read per file, e.g. 0.1 for packet level")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="balance classes: at most N rows per traffic type, e.g. 100000")
    ap.add_argument("--multiclass", action="store_true", help="predict the attack type")
    args = ap.parse_args()
    data = load_dataset(args.data_dir, args.sample, args.max_per_class)
    print(f"Total: {len(data):,} rows, {int(data['label'].sum()):,} attacks, "
          f"{data.shape[1] - 2} features\n")
    results = evaluate(data, args.multiclass)
    mode = "multiclass" if args.multiclass else "binary"
    print(f"Feature level: {args.level} | {mode}")
    print(results.round(4).to_string(index=False))
    results.to_csv(f"results_{args.level}_{mode}.csv", index=False)

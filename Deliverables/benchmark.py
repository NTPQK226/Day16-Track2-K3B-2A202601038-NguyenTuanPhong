#!/usr/bin/env python3
"""
LightGBM Benchmark - Credit Card Fraud Detection
Measures training time, inference latency/throughput, and classification metrics.
"""

import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# === Load Data ===
print("Loading dataset...")
start_time = time.time()
df = pd.read_csv('/home/ntpqk226_gmail_com/ml-benchmark/creditcard.csv')
load_time = time.time() - start_time
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns in {load_time:.2f}s")

# === Prepare Data ===
X = df.drop(['Time', 'Amount', 'Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# === Train LightGBM ===
print("\nTraining LightGBM...")
start_train = time.time()

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)

train_time = time.time() - start_train
best_iter = model.best_iteration_
print(f"Training completed in {train_time:.2f}s | Best iteration: {best_iter}")

# === Evaluate ===
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc_roc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# === Inference Latency (1 row) ===
single_row = X_test.iloc[:1]
latency_times = []
for _ in range(100):
    start = time.time()
    model.predict(single_row)
    latency_times.append((time.time() - start) * 1000)  # ms
avg_latency = np.mean(latency_times)

# === Inference Throughput (1000 rows) ===
batch = X_test.iloc[:1000]
throughput_times = []
for _ in range(10):
    start = time.time()
    model.predict(batch)
    throughput_times.append((time.time() - start) * 1000)  # ms
avg_throughput_time = np.mean(throughput_times)
throughput_per_sec = 1000 / avg_throughput_time

# === Print Results ===
print("\n" + "="*50)
print("BENCHMARK RESULTS")
print("="*50)
print(f"Load time:        {load_time:.3f} s")
print(f"Training time:    {train_time:.3f} s")
print(f"Best iteration:   {best_iter}")
print(f"AUC-ROC:          {auc_roc:.4f}")
print(f"Accuracy:         {accuracy:.4f}")
print(f"F1-Score:         {f1:.4f}")
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"Latency (1 row):  {avg_latency:.3f} ms")
print(f"Throughput:       {throughput_per_sec:.1f} rows/sec")
print("="*50)

# === Save Results ===
results = {
    "load_time_seconds": round(load_time, 3),
    "training_time_seconds": round(train_time, 3),
    "best_iteration": best_iter,
    "auc_roc": round(auc_roc, 4),
    "accuracy": round(accuracy, 4),
    "f1_score": round(f1, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "inference_latency_ms": round(avg_latency, 3),
    "inference_throughput_rows_per_sec": round(throughput_per_sec, 1)
}

with open('/home/ntpqk226_gmail_com/ml-benchmark/benchmark_result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to benchmark_result.json")

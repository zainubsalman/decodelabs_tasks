"""
DecodeLabs Project 3: Unsupervised Learning (Customer Segmentation)
======================================================================

Goal: Use distance-based algorithms to discover hidden mathematical
groupings in unlabeled retail data.

This script follows the IPO (Input-Process-Output) architecture from
the brief:
    1. SCALE     -> StandardScaler (Input)
    2. COMPRESS  -> Principal Component Analysis (Process)
    3. CLUSTER   -> K-Means, validated with Elbow Method + Silhouette Score
    4. TRANSLATE -> Reverse-engineer centroids into human-readable
                     business personas (Output)

Dataset note: no specific dataset file was named in the slides you
shared. Replace the synthetic data loader below with your real
dataset once you have it (e.g. pd.read_csv("your_customers.csv")).
The rest of the pipeline works unchanged as long as your dataset is
a table of numeric customer features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator


# ============================================================
# STEP 0: LOAD THE DATA
# ============================================================

USE_REAL_DATA = False  # set True once you have your real dataset

def load_data():
    if USE_REAL_DATA:
        df = pd.read_csv("your_customers.csv")
        return df

    # ---- Synthetic fallback (demo only) ----
    # Simulates a retail dataset with 20+ numeric behavioral columns
    # and naturally forms a few underlying customer groups.
    np.random.seed(42)
    n_customers = 600

    # Create 4 underlying "true" segments so clusters actually exist
    segment_centers = np.random.uniform(-5, 5, size=(4, 22))
    labels_hidden = np.random.choice(4, size=n_customers)
    X = np.array([
        segment_centers[label] + np.random.normal(0, 1.5, size=22)
        for label in labels_hidden
    ])

    columns = [f"feature_{i+1}" for i in range(22)]
    df = pd.DataFrame(X, columns=columns)
    return df


# ============================================================
# PHASE 1 (SCALE): STANDARDIZE BEFORE ANY DISTANCE CALCULATION
# ============================================================

def scale_features(df: pd.DataFrame):
    """
    StandardScaler puts every column on the same footing (mean 0,
    std 1), so no single feature dominates distance calculations
    just because it has bigger raw numbers.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    return X_scaled, scaler


# ============================================================
# PHASE 2 (COMPRESS): PCA DOWN TO THE 95% VARIANCE THRESHOLD
# ============================================================

def compress_with_pca(X_scaled, variance_threshold: float = 0.95):
    """
    Fit PCA keeping all components first, then find how many
    components are needed to retain at least 95% of the original
    variance (the "95% Rule" from the brief).
    """
    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= variance_threshold) + 1)

    print(f"Components needed to retain {variance_threshold*100:.0f}% variance: {n_components}")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


# ============================================================
# PHASE 3 (CLUSTER): FIND OPTIMAL K, THEN RUN K-MEANS
# ============================================================

def find_optimal_k(X_pca, k_range=range(1, 11)):
    """
    Elbow Method: compute WCSS (Within-Cluster Sum of Squares) for
    each K, then use the Kneedle Algorithm to mathematically find
    the "elbow", the point of maximum curvature where adding more
    clusters stops giving meaningful improvement.
    """
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        wcss.append(km.inertia_)

    knee = KneeLocator(list(k_range), wcss, curve="convex", direction="decreasing")
    elbow_k = knee.elbow

    print(f"WCSS by K: {[round(w, 1) for w in wcss]}")
    print(f"Elbow Method suggests K = {elbow_k}")

    return elbow_k, wcss


def confirm_k_with_silhouette(X_pca, k_range=range(2, 11)):
    """
    Silhouette Score: for each K, measures how well-separated the
    clusters are (closer to +1.0 is better, near 0.0 means clusters
    overlap). This is the second "diagnostic gatekeeper".
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, cluster_labels)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    formatted_scores = {k: round(v, 3) for k, v in scores.items()}
    print(f"Silhouette scores by K: {formatted_scores}")
    print(f"Silhouette Score suggests K = {best_k} (score: {scores[best_k]:.3f})")

    return best_k, scores


def run_kmeans(X_pca, k: int):
    """Fit the final K-Means model with the chosen number of clusters."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_pca)
    return km, cluster_labels


# ============================================================
# PHASE 4 (TRANSLATE): REVERSE-ENGINEER CENTROIDS INTO PERSONAS
# ============================================================

def translate_centroids(km: KMeans, pca: PCA, scaler: StandardScaler, feature_names):
    """
    K-Means centroids live in scaled, PCA-compressed space, meaningless
    numbers to a business stakeholder. This function reverses both
    transformations to reconstruct real, human-readable feature values
    for each cluster center.

        Step 1: inverse PCA      (compressed space -> scaled space)
        Step 2: inverse scaling  (scaled space -> original units)
    """
    centroids_pca_space = km.cluster_centers_

    # Step 1: undo PCA
    centroids_scaled_space = pca.inverse_transform(centroids_pca_space)

    # Step 2: undo StandardScaler
    centroids_original_space = scaler.inverse_transform(centroids_scaled_space)

    centroid_df = pd.DataFrame(centroids_original_space, columns=feature_names)
    centroid_df.index.name = "Cluster"
    return centroid_df


def build_persona_summary(df_original: pd.DataFrame, cluster_labels, feature_names):
    """
    Build a persona summary using the ACTUAL customer rows grouped by
    cluster (mean of each feature per cluster), which is usually more
    faithful to the real data than the reconstructed centroid alone.
    """
    df_with_clusters = df_original.copy()
    df_with_clusters["Cluster"] = cluster_labels

    persona_summary = df_with_clusters.groupby("Cluster")[feature_names].mean()
    persona_summary["Customer_Count"] = df_with_clusters.groupby("Cluster").size()
    return persona_summary


# ============================================================
# FULL PIPELINE
# ============================================================

def run_customer_segmentation():
    print("Step 1 (SCALE): Loading and standardizing data...")
    df = load_data()
    feature_names = df.columns.tolist()
    X_scaled, scaler = scale_features(df)

    print("\nStep 2 (COMPRESS): Applying PCA...")
    X_pca, pca = compress_with_pca(X_scaled)

    print("\nStep 3a (CLUSTER): Running Elbow Method...")
    elbow_k, wcss = find_optimal_k(X_pca)

    print("\nStep 3b (CLUSTER): Confirming with Silhouette Score...")
    silhouette_k, scores = confirm_k_with_silhouette(X_pca)

    # Prefer the silhouette-confirmed K when the two methods disagree,
    # since it directly measures cluster separation quality.
    optimal_k = silhouette_k
    print(f"\nFinal chosen K = {optimal_k} (Elbow suggested {elbow_k}, Silhouette confirmed {silhouette_k})")

    print(f"\nStep 3c: Fitting final K-Means with K = {optimal_k}...")
    km, cluster_labels = run_kmeans(X_pca, optimal_k)

    print("\nStep 4 (TRANSLATE): Reverse-engineering centroids...")
    centroid_df = translate_centroids(km, pca, scaler, feature_names)
    print("\nReconstructed cluster centroids (original feature units):")
    print(centroid_df.round(2))

    print("\nBusiness Persona Summary (actual customer averages per cluster):")
    persona_df = build_persona_summary(df, cluster_labels, feature_names)
    print(persona_df.round(2))

    return persona_df


if __name__ == "__main__":
    run_customer_segmentation()
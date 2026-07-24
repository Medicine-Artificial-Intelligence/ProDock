import os
import argparse
import pandas as pd
import umap
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, SDMolSupplier, MolToSmiles
from rdkit.ML.Cluster import Butina
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
from joblib import Parallel, delayed
from natsort import natsorted
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def extract_smiles_from_sdf_folder(sdf_folder, output_path):
    all_data = []
    for filename in os.listdir(sdf_folder):
        if filename.lower().endswith(".sdf"):
            path = os.path.join(sdf_folder, filename)
            suppl = SDMolSupplier(path)
            for mol in suppl:
                if mol is None:
                    continue
                name = filename[:-4]
                smi = MolToSmiles(mol)
                all_data.append({"Compounds": name, "Smiles": smi})
    df_sdf = pd.DataFrame(all_data)
    df_sdf.to_csv(output_path, index=False)
    print(f"[✓] Extracted {len(df_sdf)} SMILES to: {output_path}")
    return df_sdf


def convert_single_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return fp, mol
    return None, None


def smiles_to_ecfp4(smiles_list, n_jobs):
    results = Parallel(n_jobs=n_jobs)(delayed(convert_single_smiles)(smi) for smi in smiles_list)
    fps, mols = zip(*results)
    return list(fps), list(mols)


def tanimoto_distance_matrix(fps, n_jobs):
    def get_dists(i):
        return [1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]

    all_dists = Parallel(n_jobs=n_jobs)(delayed(get_dists)(i) for i in range(1, len(fps)))
    return [d for sublist in all_dists for d in sublist]


def butina_clustering(fps, cutoff, n_jobs):
    dist_cutoff = 1.0 - cutoff
    dists = tanimoto_distance_matrix(fps, n_jobs=n_jobs)
    clusters = Butina.ClusterData(dists, len(fps), dist_cutoff, isDistData=True)
    return clusters


def compute_centroids(clusters, fps, compounds):
    centroids = []
    for cluster in clusters:
        if len(cluster) == 1:
            centroids.append(cluster[0])
        else:
            max_sim = -1
            center = cluster[0]
            for i in cluster:
                sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in cluster if i != j]
                avg_sim = sum(sims) / len(sims) if sims else 0
                if avg_sim > max_sim:
                    max_sim = avg_sim
                    center = i
            centroids.append(center)
    return centroids


def compute_silhouette(fps, labels):
    arr = []
    for fp in fps:
        a = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, a)
        arr.append(a)
    arr = np.array(arr)
    if len(set(labels)) < 2 or len(arr) <= len(set(labels)):
        return -1
    score = silhouette_score(arr, labels)
    return score


def assign_remaining_by_centroid(
    fps, clusters, compounds, smiles_list, min_compounds, tanimoto_threshold, threshold_decrement, max_iter=10
):
    large_clusters = [c for c in clusters if len(c) >= min_compounds]
    small_clusters = [c for c in clusters if len(c) < min_compounds]

    unassigned = [i for cluster in small_clusters for i in cluster]
    cluster_assignments = {i: list(cluster) for i, cluster in enumerate(large_clusters)}
    cluster_ids = list(cluster_assignments.keys())

    current_threshold = tanimoto_threshold

    for iteration in range(max_iter):
        print(f"\n[→] Iteration {iteration + 1} | Threshold: {current_threshold:.4f}")

        centroids = {}
        for cid in cluster_ids:
            cluster = cluster_assignments[cid]
            centroid = compute_centroids([cluster], fps, compounds)[0]
            centroids[cid] = centroid

        newly_assigned = []
        still_unassigned = []

        for idx in unassigned:
            best_cid = None
            best_sim = -1
            for cid, c_idx in centroids.items():
                sim = DataStructs.TanimotoSimilarity(fps[idx], fps[c_idx])
                if sim > best_sim:
                    best_sim = sim
                    best_cid = cid

            if best_sim >= current_threshold:
                cluster_assignments[best_cid].append(idx)
                newly_assigned.append(idx)
            else:
                still_unassigned.append(idx)

        print(f"[✓] Assigned {len(newly_assigned)} compounds in this round")

        if not newly_assigned:
            print("[✓] No new assignments — converged.")
            break

        unassigned = still_unassigned
        current_threshold -= threshold_decrement
        if current_threshold <= 0:
            print("[!] Threshold too low, stopping assignment.")
            break

    next_id = max(cluster_ids) + 1
    for idx in unassigned:
        cluster_assignments[next_id] = [idx]
        next_id += 1

    final_clusters = list(cluster_assignments.values())
    return final_clusters


def visualize_clusters(fps, clusters, output_prefix):
    # === Convert fingerprints to array ===
    arr = []
    for fp in fps:
        a = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, a)
        arr.append(a)
    arr = np.array(arr)

    # === Cluster labels ===
    labels = np.zeros(len(fps), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cluster_id

    # === Original space silhouette ===
    sil_ecfp4 = compute_silhouette(fps, labels)
    sil_ecfp4_str = f"{sil_ecfp4:.2f}" if sil_ecfp4 >= 0 else "NA"

    # === Step 1: PCA ===
    pca = PCA(n_components=50)
    reduced_pca = pca.fit_transform(arr)

    # === Step 2a: t-SNE ===
    tsne = TSNE(n_components=2, perplexity=70, max_iter=2000, random_state=42)
    reduced_tsne = tsne.fit_transform(reduced_pca)

    df_tsne = pd.DataFrame(reduced_tsne, columns=["tsne-1", "tsne-2"])
    df_tsne["Cluster"] = labels.astype(int)

    sil_tsne = silhouette_score(reduced_tsne, labels) if len(set(labels)) > 1 else -1
    sil_tsne_str = f"{sil_tsne:.2f}" if sil_tsne >= 0 else "NA"

    num_clusters = len(set(labels))
    palette = sns.color_palette("hls", n_colors=num_clusters)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_tsne, x="tsne-1", y="tsne-2", hue="Cluster", palette=palette, s=8, legend="full")
    cluster_sizes = [len(cluster) for cluster in clusters]
    legend_labels = [f"Cluster {i} (n={size})" for i, size in enumerate(cluster_sizes)]
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=legend_labels,
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        ncol=2,
        fontsize="small",
        title_fontsize="medium",
        frameon=True,
    )
    plt.title("t-SNE Projection of Clusters")
    plt.subplots_adjust(right=0.75)
    tsne_path = (
        f"{output_prefix}_TSNE_silEcfp4_{sil_ecfp4_str}_silTSNE_{sil_tsne_str}"
        f"_decrement{args.threshold_decrement}_tanimoto{args.tanimoto_threshold}"
        f"_min{args.min_compounds}.png"
    )
    plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === Step 2b: UMAP ===
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="euclidean", random_state=42)
    reduced_umap = reducer.fit_transform(reduced_pca)

    df_umap = pd.DataFrame(reduced_umap, columns=["umap-1", "umap-2"])
    df_umap["Cluster"] = labels.astype(int)

    sil_umap = silhouette_score(reduced_umap, labels) if len(set(labels)) > 1 else -1
    sil_umap_str = f"{sil_umap:.2f}" if sil_umap >= 0 else "NA"

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_umap, x="umap-1", y="umap-2", hue="Cluster", palette=palette, s=8, legend="full")
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=legend_labels,
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        ncol=2,
        fontsize="small",
        title_fontsize="medium",
        frameon=True,
    )
    plt.title("UMAP Projection of Clusters")
    plt.subplots_adjust(right=0.75)
    umap_path = (
        f"{output_prefix}_UMAP_silEcfp4_{sil_ecfp4_str}_silUMAP_{sil_umap_str}"
        f"_decrement{args.threshold_decrement}_tanimoto{args.tanimoto_threshold}"
        f"_min{args.min_compounds}.png"
    )
    plt.savefig(umap_path, dpi=300, bbox_inches="tight")
    plt.close()

    return labels.tolist(), sil_ecfp4


def export_centroid_similarity(centroid_indices, fps, compounds, output_path):
    centroid_names = [compounds[i] for i in centroid_indices]
    centroid_fps = [fps[i] for i in centroid_indices]

    sim_matrix = []
    for i in range(len(centroid_fps)):
        row = []
        for j in range(len(centroid_fps)):
            sim = DataStructs.TanimotoSimilarity(centroid_fps[i], centroid_fps[j])
            row.append(sim)
        sim_matrix.append(row)

    df_sim = pd.DataFrame(sim_matrix, index=centroid_names, columns=centroid_names)
    df_sim.to_csv(output_path)
    print(f"[✓] Saved centroid similarity matrix to: {output_path}")


def plot_heatmap(csv_path, output_path):
    # Load CSV with first column as index
    df = pd.read_csv(csv_path, index_col=0)

    # Natural sort of rows and columns
    sorted_index = natsorted(df.index)
    df = df.loc[sorted_index, sorted_index]

    # Convert to float
    df = df.astype(float)

    # Mask out 1.0 similarity
    df_masked = df.mask(df == 1.0)

    # Plot heatmap
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        df_masked,
        cmap="viridis",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        cbar_kws={"label": "Tanimoto Similarity"},
        mask=df == 1.0,
    )
    plt.title("Pairwise Tanimoto Similarity Heatmap (Natural Sorted)", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap saved to {output_path}")


def main(args):
    smiles_csv_path = os.path.join(args.output_dir, "sdf_extracted.csv")
    df_sdf = extract_smiles_from_sdf_folder(args.ligand_dir, smiles_csv_path)

    df_src = pd.read_csv(args.source_csv)
    df = pd.merge(df_src, df_sdf, on="Compounds", how="inner")
    print(f"[✓] Merged {len(df)} compounds from source CSV and ligand folder")

    compounds = df["Compounds"].tolist()
    smiles_list = df["Smiles"].tolist()
    fps, mols = smiles_to_ecfp4(smiles_list, n_jobs=args.cpu)
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    fps = [fps[i] for i in valid_indices]
    compounds = [compounds[i] for i in valid_indices]
    smiles_list = [smiles_list[i] for i in valid_indices]

    clusters = butina_clustering(fps, cutoff=args.tanimoto_threshold, n_jobs=args.cpu)

    clusters = assign_remaining_by_centroid(
        fps,
        clusters,
        compounds,
        smiles_list,
        min_compounds=args.min_compounds,
        tanimoto_threshold=args.tanimoto_threshold,
        threshold_decrement=args.threshold_decrement,
    )

    compound_cluster_mapping = []
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            compound_cluster_mapping.append(
                {"Compounds": compounds[idx], "Smiles": smiles_list[idx], "Cluster": cluster_id}
            )

    centroids = compute_centroids(clusters, fps, compounds)
    print(f"\n[✓] Total clusters: {len(clusters)}")
    print("[✓] Centroid compounds:")
    for idx in centroids:
        print(f" - {compounds[idx]}")

    plot_prefix = os.path.join(args.output_dir, "cluster_plot")
    cluster_labels, silhouette = visualize_clusters(fps, clusters, output_prefix=plot_prefix)

    output_cluster_csv = os.path.join(
        args.output_dir,
        f"compound_clusters_decrement{args.threshold_decrement}"
        f"_tanimoto{args.tanimoto_threshold}_min{args.min_compounds}.csv",
    )
    df_result = pd.DataFrame({"Compounds": compounds, "Smiles": smiles_list, "Cluster": cluster_labels})
    df_result.to_csv(output_cluster_csv, index=False)
    print(f"[✓] Exported cluster assignments to: {output_cluster_csv}")

    output_sim_csv = os.path.join(
        args.output_dir,
        f"centroid_similarity_decrement{args.threshold_decrement}"
        f"_tanimoto{args.tanimoto_threshold}_min{args.min_compounds}.csv",
    )
    export_centroid_similarity(centroids, fps, compounds, output_sim_csv)

    output_full_cluster_csv = os.path.join(
        args.output_dir,
        f"all_compound_clusters_decrement{args.threshold_decrement}"
        f"_tanimoto{args.tanimoto_threshold}_min{args.min_compounds}.csv",
    )
    df_map = pd.DataFrame(compound_cluster_mapping)
    df_map.to_csv(output_full_cluster_csv, index=False)
    print(f"[✓] Exported full compound-cluster list to: {output_full_cluster_csv}")

    # Save silhouette to txt
    score_txt_path = os.path.join(args.output_dir, "silhouette_score.txt")
    with open(score_txt_path, "w") as f:
        f.write(f"Silhouette Coefficient: {silhouette:.4f}\n")
    print(f"[✓] Silhouette score saved to: {score_txt_path}")
    plot_heatmap(
        f"{args.output_dir}/centroid_similarity_decrement{args.threshold_decrement}"
        f"_tanimoto{args.tanimoto_threshold}_min{args.min_compounds}.csv",
        f"{args.output_dir}/centroid_similarity_decrement{args.threshold_decrement}"
        f"_tanimoto{args.tanimoto_threshold}_min{args.min_compounds}.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Butina Clustering with Distance Matrix and Silhouette Scoring")
    parser.add_argument("--ligand_dir", required=True, help="Directory of .sdf files")
    parser.add_argument("--output_dir", required=True, help="Directory to store output CSVs and plots")
    parser.add_argument("--source_csv", required=True, help="CSV file containing Compounds column")
    parser.add_argument("--tanimoto_threshold", type=float, default=0.35, help="Initial Tanimoto similarity cutoff")
    parser.add_argument(
        "--threshold_decrement", type=float, default=0.05, help="Decrease of similarity cutoff per iteration"
    )
    parser.add_argument("--cpu", type=int, default=-1, help="Number of CPUs to use for parallel jobs")
    parser.add_argument("--min_compounds", type=int, default=10, help="Minimum cluster size before reassignment")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)

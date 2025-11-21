import argparse
import ast
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(1, os.getcwd())
from misc.general_utils import set_seed_reproducability, get_device, get_model
from misc.log_utils import Logger

def main(exp_name,
         pretrained_weight_path=None,
         model_c=48,
         model_nof_joints=17,
         model_bn_momentum=0.1,
         seed=1,
         device=None,
         model_name = 'hrnet'
         ):

    # Seeds
    set_seed_reproducability(seed)

    device = get_device(device)

    print("\nStarting experiment `%s` @ %s\n" % (exp_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    os.makedirs(exp_name, 0o755, exist_ok=True)  # exist_ok=False to avoid overwriting    

    sys.stdout = Logger("{}/test_model_weights.log".format(exp_name))
    # sys.stderr = sys.stdout
    command_line_args = sys.argv
    command = " ".join(command_line_args)
    print(f"The command that ran this script: {command}")

    model = get_model(model_name=model_name, model_c=model_c, model_nof_joints=model_nof_joints,
        model_bn_momentum=model_bn_momentum, device=device, pretrained_weight_path=pretrained_weight_path)
    
    joint_prototypes = model.get_joint_prototypes()
    
    prototypes = joint_prototypes.weight.data.cpu().numpy().squeeze()
    bias = joint_prototypes.bias.data.cpu().numpy()
    
    import datasets.CustomDS.data_configs.COCO_configs as coco_configs
    
    for joint_index, joint_info in coco_configs.COCO_dataset_info['keypoint_info'].items():
        print(joint_index, joint_info['name'])
    # name = coco_configs.COCO_dataset_info['keypoint_info'][joint_index]['name']

    print(prototypes.shape)
    print(bias)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.stats import pearsonr

    def mask_diagonal(matrix):
        m = matrix.copy()
        np.fill_diagonal(m, np.nan)
        return m

    # ---------------------------------------------------------------
    # Load keypoint names
    # ---------------------------------------------------------------
    names = [coco_configs.COCO_dataset_info['keypoint_info'][i]['name']
            for i in range(len(prototypes))]


    # Helper function to save and close figures
    def savefig(name):
        path = os.path.join(exp_name, name)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


    # ===============================================================
    # 1. COSINE SIMILARITY MATRIX
    # ===============================================================
    cos_sim = cosine_similarity(prototypes)
    cos_sim = mask_diagonal(cos_sim)

    plt.figure(figsize=(10, 8))
    plt.imshow(cos_sim, cmap="viridis")
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(17), names, rotation=90)
    plt.yticks(range(17), names)
    plt.title("Cosine Similarity Between Keypoint Prototypes")
    savefig("cosine_similarity.png")


    # ===============================================================
    # 2. EUCLIDEAN DISTANCE MATRIX
    # ===============================================================
    eucl_dist = euclidean_distances(prototypes)
    eucl_dist = mask_diagonal(eucl_dist)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(eucl_dist, cmap="viridis")
    plt.colorbar(label="Euclidean Distance")
    plt.xticks(range(17), names, rotation=90)
    plt.yticks(range(17), names)
    plt.title("Euclidean Distance Between Keypoint Prototypes")
    savefig("euclidean_distance.png")


    # ===============================================================
    # 3. EUCLIDEAN NORM BAR PLOT
    # ===============================================================
    norms = np.linalg.norm(prototypes, axis=1)

    plt.figure(figsize=(10, 5))
    plt.bar(range(17), norms)
    plt.xticks(range(17), names, rotation=90)
    plt.ylabel("L2 Norm")
    plt.title("Norm of Each Keypoint Prototype")
    savefig("prototype_norms.png")


    # ===============================================================
    # 4. PCA 2-D PROJECTION
    # ===============================================================
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(prototypes)

    plt.figure(figsize=(7, 6))
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1])
    for i, name in enumerate(names):
        plt.text(pca_2d[i, 0], pca_2d[i, 1], name)
    plt.title("PCA Projection (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig("pca_2d.png")


    # ===============================================================
    # 5. t-SNE 2-D PROJECTION
    # ===============================================================
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=50, n_iter=4000)
    tsne_2d = tsne.fit_transform(prototypes)

    plt.figure(figsize=(7, 6))
    plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1])
    for i, name in enumerate(names):
        plt.text(tsne_2d[i, 0], tsne_2d[i, 1], name)
    plt.title("t-SNE Embedding")
    savefig("tsne_2d.png")


    # ===============================================================
    # 6. HIERARCHICAL CLUSTERING + DENDROGRAM
    # ===============================================================
    Z = linkage(prototypes, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.title("Hierarchical Clustering (Ward)")
    savefig("dendrogram.png")


    # ===============================================================
    # 7. ANGLE MATRIX
    # ===============================================================
    angles = np.arccos(np.clip(cos_sim, -1, 1))
    angles = mask_diagonal(angles)

    plt.figure(figsize=(10, 8))
    plt.imshow(angles, cmap="viridis")
    plt.colorbar(label="Angle (radians)")
    plt.xticks(range(17), names, rotation=90)
    plt.yticks(range(17), names)
    plt.title("Angular Distance")
    savefig("angle_matrix.png")


    # ===============================================================
    # 8. NORM vs BIAS
    # ===============================================================
    plt.figure(figsize=(6, 5))
    plt.scatter(norms, bias)
    for i, name in enumerate(names):
        plt.text(norms[i], bias[i], name)
    plt.xlabel("Prototype Norm")
    plt.ylabel("Bias")
    plt.title("Bias vs Prototype Norm")
    savefig("bias_vs_norm.png")


    # ===============================================================
    # 9. NEAREST NEIGHBORS
    # ===============================================================
    with open(os.path.join(exp_name, "nearest_neighbors.txt"), "w") as f:
        f.write("Nearest Neighbors (Cosine Similarity)\n\n")
        for i in range(17):
            sims = cos_sim[i].copy()
            sims[i] = -np.inf
            nn = sims.argmax()
            f.write(f"{names[i]:20s} --> {names[nn]} (cos={sims[nn]:.4f})\n")


    # ===============================================================
    # 10. DIMENSION VARIANCE
    # ===============================================================
    dim_var = np.var(prototypes, axis=0)
    top_dims = dim_var.argsort()[::-1][:10]

    with open(os.path.join(exp_name, "dimension_variance.txt"), "w") as f:
        f.write("Top 10 Most Varying Feature Dimensions\n\n")
        for d in top_dims:
            f.write(f"Dim {d}: Var = {dim_var[d]:.6f}\n")


    # ===============================================================
    # 11. TOP-k DIMENSIONS PER PROTOTYPE
    # ===============================================================
    top_k = 5
    with open(os.path.join(exp_name, "top_dims_per_prototype.txt"), "w") as f:
        f.write("Top-5 Abs-Value Dimensions per Prototype\n\n")
        for i in range(17):
            idx = np.argsort(np.abs(prototypes[i]))[::-1][:top_k]
            f.write(f"{names[i]:20s} --> {idx.tolist()}\n")


    # ===============================================================
    # 12. BIAS-AWARE EXTENDED VECTOR SIMILARITY
    # ===============================================================
    extended = np.concatenate([prototypes, bias[:, None]], axis=1)
    ext_cos_sim = cosine_similarity(extended)
    ext_cos_sim = mask_diagonal(ext_cos_sim)

    plt.figure(figsize=(10, 8))
    plt.imshow(ext_cos_sim, cmap="viridis")
    plt.colorbar(label="Cosine Similarity (Bias-aware)")
    plt.xticks(range(17), names, rotation=90)
    plt.yticks(range(17), names)
    plt.title("Bias-Aware Similarity (Prototypes + Bias)")
    savefig("bias_aware_similarity.png")


    # ===============================================================
    # 13. BIAS CORRELATIONS
    # ===============================================================
    corr_norm_bias, p1 = pearsonr(norms, bias)
    corr_meanw_bias, p2 = pearsonr(prototypes.mean(axis=1), bias)

    with open(os.path.join(exp_name, "bias_correlations.txt"), "w") as f:
        f.write("Bias Correlations\n\n")
        f.write(f"Corr(norm, bias) = {corr_norm_bias:.4f} (p={p1:.4f})\n")
        f.write(f"Corr(mean(weight dims), bias) = {corr_meanw_bias:.4f} (p={p2:.4f})\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=48)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    parser.add_argument("--model_name", help="poseresnet or hrnet", type=str, default='hrnet')
    
    args = parser.parse_args()

    
    main(**args.__dict__)

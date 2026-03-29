import numpy as np
import argparse
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
import json
import torch
import pickle

def calculate_exact_budgets(total_budget: int, weights: np.ndarray) -> np.ndarray:
    """
    使用最大余额法 (Largest Remainder Method) 分配 PCA 主成分的数据配额，
    确保所有分配的配额总和绝对等于 total_budget，避免 round() 造成的名额丢失或溢出。
    """
    exact_budgets = total_budget * weights
    budgets = np.floor(exact_budgets).astype(int)
    remainder = total_budget - np.sum(budgets)
    
    if remainder > 0:
        fractional_parts = exact_budgets - budgets
        largest_remainder_indices = np.argsort(fractional_parts)[::-1]
        for i in range(int(remainder)):
            budgets[largest_remainder_indices[i]] += 1
            
    return budgets

def load_validation_components(gradients_path: str, val_k: float, transpose: bool, pca_mode: str):
    """
    Step 2: Extract Core Knowledge from Validation Set
    使用 PCA 提取核心知识并返回主成分向量及权重。
    """
    data = torch.load(gradients_path)
    if torch.is_tensor(data):
        data = data.numpy()
        
    if transpose:
        data = data.T
        
    print(f"验证集梯度加载完成，Shape: {data.shape}")

    if pca_mode == "variance":
        pca = PCA(n_components=val_k)
        pca.fit(data)
        components = pca.components_
        weights = pca.explained_variance_ratio_
        print(f"[PCA - Variance Mode] 提取了 {len(weights)} 个主成分，截取了前 {val_k*100}% 的解释方差。")
    else:
        n_components_to_keep = max(1, int(data.shape[1] * val_k))
        pca = PCA(n_components=n_components_to_keep)
        pca.fit(data)
        components = pca.components_
        weights = pca.explained_variance_ratio_
        print(f"[PCA - Count Ratio Mode] 提取了前 {n_components_to_keep} 个主成分，覆盖了 {np.sum(weights)*100:.2f}% 的方差。")
        
    weights = weights / np.sum(weights) 
    return components, weights

def load_train_gradients(gradients_path: str, transpose: bool):
    data = torch.load(gradients_path)
    if torch.is_tensor(data):
        data = data.numpy()  
    
    if transpose:
        data = data.T
        
    target_number = len(data)
    print(f"训练集梯度加载完成，Shape: {data.shape}，共 {target_number} 条数据。")
    return data, target_number

class GraphWalk:
    """
    Core implementation of the Gradient Walk Algorithm (Algorithm 1, Appendix C).
    
    [CRITICAL INPUT ASSUMPTION - Knowledge Coherence (Eq. 4)]: 
    The `graph` parameter MUST be a pre-sorted adjacency list. 
    For any given node i, `graph[i]` must contain its neighbors sorted in strictly 
    descending order of their cosine similarity to node i. 
    This sorting is the mathematical foundation for achieving the `argmax` 
    (Knowledge Coherence) in Eq.4 via early-stopping (taking the first valid neighbor).
    Passing an unsorted graph will lead to mathematically invalid selections.
    
    [FALLBACK LOGIC]:
    When the local graph walk is exhausted (no valid neighbors maintain coherence 
    without violating constraints), the algorithm strictly follows the fallback 
    condition in Appendix C. It abandons the local anchor s* and performs a global 
    search, re-anchoring to the sample most similar to the core knowledge K_v.
    """
    def __init__(
        self,
        training_data_gradients: np.ndarray,
        validation_gradients: np.ndarray,
        weight: np.ndarray,
        graph: np.ndarray,
        validation_topk: np.ndarray,
        total: int,
    ):
        self.training_data_gradients = training_data_gradients
        self.validation_gradients = validation_gradients
        self.total = total
        self.graph = graph
        self.weight = weight
        self.validation_topk = validation_topk

    def graph_walk(self, train_k: float, way: float) -> list:
        target_number = int(self.total * train_k)
        
        # 使用最大余额法，确保名额绝对精确
        budgets = calculate_exact_budgets(target_number, self.weight)

        global_excluded: set = set()
        all_selected: list = []

        with tqdm(total=target_number, desc="Gradient Walking", unit="sample") as pbar:
            for i, budget in enumerate(budgets):
                if budget <= 0:
                    continue

                current_set: list = []
                kv = self.validation_gradients[i]
                
                # 预先计算所有训练样本与当前核心知识 kv 的相似度
                all_sims = np.dot(self.training_data_gradients, kv)
                masked_sims = all_sims.copy()

                # ---- 初始化 Anchor ----
                anchor = -1
                for candidate in self.validation_topk[i]:
                    candidate = int(candidate)
                    if candidate not in global_excluded:
                        anchor = candidate
                        break

                if anchor == -1:
                    # Anchor 全局搜索
                    if global_excluded:
                        masked_sims[list(global_excluded)] = -np.inf
                    anchor = int(np.argmax(masked_sims))
                    if masked_sims[anchor] == -np.inf:
                        break # 数据集已耗尽

                current_set.append(anchor)
                global_excluded.add(anchor)
                all_selected.append(anchor)
                pbar.update(1)

                # 维护梯度累加和，将后续计算时间复杂度降至 O(D)
                sum_grad_S_prime = self.training_data_gradients[anchor].copy()

                # ---- Graph Walk ----
                while len(current_set) < budget:
                    last_node = current_set[-1]
                    neighbors = self.graph[last_node]

                    norm_old = np.linalg.norm(sum_grad_S_prime)
                    old_sim = abs(np.dot(sum_grad_S_prime / norm_old, kv)) if norm_old > 1e-10 else 0.0

                    found = False
                    
                    # 1. 局部图游走
                    for neighbor in neighbors:
                        neighbor = int(neighbor)
                        if neighbor in global_excluded:
                            continue

                        # Constraint 5: 无冲突检查
                        conflicts = np.dot(self.training_data_gradients[current_set], self.training_data_gradients[neighbor])
                        if np.any(conflicts < 0):
                            continue

                        # Constraint 6: 核心知识一致性检查
                        new_sum_grad = sum_grad_S_prime + self.training_data_gradients[neighbor]
                        norm_new = np.linalg.norm(new_sum_grad)
                        new_sim = abs(np.dot(new_sum_grad / norm_new, kv)) if norm_new > 1e-10 else 0.0

                        if new_sim >= way * old_sim:
                            current_set.append(neighbor)
                            global_excluded.add(neighbor)
                            all_selected.append(neighbor)
                            sum_grad_S_prime = new_sum_grad
                            pbar.update(1)
                            found = True
                            break

                    # 2. 全局回退 (Fallback) 强制验证约束条件
                    if not found:
                        masked_sims = all_sims.copy()
                        if global_excluded:
                            masked_sims[list(global_excluded)] = -np.inf
                            
                        sorted_global_candidates = np.argsort(masked_sims)[::-1]
                        
                        fallback_success = False
                        for best_candidate in sorted_global_candidates:
                            best_candidate = int(best_candidate)
                            if masked_sims[best_candidate] == -np.inf:
                                break

                            # 必须满足无冲突约束
                            conflicts = np.dot(self.training_data_gradients[current_set], self.training_data_gradients[best_candidate])
                            if np.any(conflicts < 0):
                                continue

                            # 必须满足一致性约束
                            new_sum_grad = sum_grad_S_prime + self.training_data_gradients[best_candidate]
                            norm_new = np.linalg.norm(new_sum_grad)
                            new_sim = abs(np.dot(new_sum_grad / norm_new, kv)) if norm_new > 1e-10 else 0.0

                            if new_sim >= way * old_sim:
                                current_set.append(best_candidate)
                                global_excluded.add(best_candidate)
                                all_selected.append(best_candidate)
                                sum_grad_S_prime = new_sum_grad
                                pbar.update(1)
                                fallback_success = True
                                break
                                
                        if not fallback_success:
                            break 

                if len(current_set) < budget:
                    tqdm.write(f"[Warning] 主成分 {i} 预算为 {budget}，受限于严格约束，仅选出 {len(current_set)} 个样本。")

        print(f"\n[Summary] 目标计划选择样本数: {target_number}")
        print(f"[Summary] 实际成功筛选样本数: {len(all_selected)}")
        if len(all_selected) < target_number:
            print("[Notice] 实际数量小于目标数量，算法已严格执行无冲突与一致性约束，宁缺毋滥。")

        return all_selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G2IS: Gradient-based Graph Instruction Selection")
    parser.add_argument("--train_gradients_file", required=True, help="Path to training data gradients")
    parser.add_argument("--validation_gradients_file", required=True, help="Path to validation data gradients")
    parser.add_argument("--graph", required=True, help="Path to precomputed similarity graph numpy file")
    
    # 严谨的显式控制参数
    parser.add_argument("--transpose_train", action="store_true", help="显式转置训练集梯度矩阵")
    parser.add_argument("--transpose_val", action="store_true", help="显式转置验证集梯度矩阵")
    parser.add_argument("--pca_mode", type=str, choices=["variance", "count_ratio"], default="variance", 
                        help="'variance'指截取累计方差(如0.5为50%方差); 'count_ratio'指截取主成分个数比例")
    
    parser.add_argument("--val_k", type=float, default=0.5, help="PCA extraction threshold")
    parser.add_argument("--train_k", type=float, default=0.01, help="Percentage of training data to select")
    parser.add_argument("--ways", type=float, default=0.8, help="Threshold delta for validation consistency")
    parser.add_argument("--train_data_dir", required=True, help="Path to original training data (JSONL)")
    parser.add_argument("--save", required=True, help="Directory to save the selected subset")
    args = parser.parse_args()
    
    # 1. 加载数据
    graph = np.load(args.graph)  
    train_gradients, number = load_train_gradients(args.train_gradients_file, args.transpose_train)
    val_components, weights = load_validation_components(args.validation_gradients_file, args.val_k, args.transpose_val, args.pca_mode)
    
    # ---- 铁血校验与防御性编程 ----
    assert graph.shape[0] == train_gradients.shape[0], \
        f"[Error] 图节点数 ({graph.shape[0]}) 与训练样本数 ({train_gradients.shape[0]}) 不匹配！"
    assert val_components.shape[1] == train_gradients.shape[1], \
        f"[Error] 验证集梯度特征维度 ({val_components.shape[1]}) 与训练集 ({train_gradients.shape[1]}) 不一致！"
    assert 0 < args.train_k <= 1.0, f"[Error] train_k 必须在 (0, 1] 之间"
    assert 0 < args.ways <= 1.0, f"[Error] ways (delta) 必须在 (0, 1] 之间"

    # 2. 向量 L2 归一化 (大幅加速后续 Cosine Similarity 的计算)
    train_norms = np.linalg.norm(train_gradients, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1e-10 
    train_normalized = train_gradients / train_norms 
    del train_gradients 

    val_norms = np.linalg.norm(val_components, axis=1, keepdims=True)
    val_norms[val_norms == 0] = 1e-10
    val_normalized = val_components / val_norms
    
    # 3. 计算用于寻找 Anchor 的 Top-K 索引
    corr = np.dot(val_normalized, train_normalized.T) 
    
    top_k = min(1000, corr.shape[1])
    topk_indices = np.argpartition(corr, -top_k, axis=1)[:, -top_k:]
    for i in range(corr.shape[0]):
        row_topk = topk_indices[i]
        sorted_order = np.argsort(corr[i, row_topk])[::-1]
        topk_indices[i] = row_topk[sorted_order]
    del corr

    # 4. 初始化并执行游走算法
    gw = GraphWalk(
        training_data_gradients=train_normalized,
        validation_gradients=val_normalized,
        weight=weights,
        graph=graph,
        validation_topk=topk_indices,
        total=number
    )
    
    target_sample_indices = gw.graph_walk(train_k=args.train_k, way=args.ways)
    
    # 5. 保存结果 (流式读取防 OOM)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    with open(os.path.join(args.save, "target_sample.kpl"), 'wb') as f:
        pickle.dump(target_sample_indices, f)
        
    train_data = []
    with open(args.train_data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line.strip()))
                
    train_data = np.array(train_data, dtype=object)
    target_sample_indices = np.array(target_sample_indices)
    target_data = train_data[target_sample_indices]
    
    with open(os.path.join(args.save, "target.jsonl"), 'w', encoding='utf-8') as f:
        for d in target_data:
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            
    with open(os.path.join(args.save, "target.kpl"), 'wb') as f:
        pickle.dump(target_data.tolist(), f)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGraphFusion(nn.Module):
    def __init__(self, graph_dim=25, hidden_dim=64, top_k=5, ratio=0.5):
        super().__init__()
        self.semantic_graphs = nn.Parameter(torch.randn(120, graph_dim, graph_dim), requires_grad=False)  # frozen
        self.ratio = ratio
        self.query_proj = nn.Linear(graph_dim * graph_dim, hidden_dim)
        self.key_proj = nn.Linear(graph_dim * graph_dim, hidden_dim)

    def forward(self, logits, label=None):  # logits: [N, 120]
        N = logits.shape[0]  # batch size
        fused_graphs = []  #空列表

        for b in range(N):  # 遍历每个样本
            logit_b = logits[b]  # 第 i 个样本的 logits [120]
            max_score = logit_b.max()  # 获取最大分数
            mask = logit_b > self.ratio * max_score  # [120]  # 计算掩码，选择大于阈值的 logits

            selected_idx = torch.nonzero(mask).squeeze(1)  # [K]  # 获取满足条件的索引

            if torch.rand(1).item() < 0.2:  # 20%的概率
                if label is not None:
                    # Ensure the label is included in the selected indices
                    label_idx = torch.tensor(label[b], device=logit_b.device)
                    if label_idx not in selected_idx:
                        selected_idx = torch.cat([selected_idx, label_idx.unsqueeze(0)])
            selected_graphs = self.semantic_graphs[selected_idx]  # [K, 25, 25]

            if selected_graphs.shape[0] == 0:
                # fallback: use top-1
                selected_idx = logit_b.argmax().unsqueeze(0)
                selected_graphs = self.semantic_graphs[selected_idx]

            K = selected_graphs.shape[0]

            flat_graphs = selected_graphs.view(K, -1)  # [K, 625]
            query = self.query_proj(flat_graphs)  # [K, hidden]
            key = self.key_proj(flat_graphs)      # [K, hidden]

            attn_score = torch.softmax((query @ key.T).mean(dim=1), dim=0)  # [K]
            fused = torch.sum(attn_score.view(K, 1, 1) * selected_graphs, dim=0)  # [25,25]
            fused_graphs.append(fused)

        A_fused = torch.stack(fused_graphs, dim=0)  # [N, 25, 25]
        return A_fused

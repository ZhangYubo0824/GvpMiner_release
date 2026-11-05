"""
M2: ESM-2 蛋白质语言模型嵌入
=============================
使用 Meta AI 的 ESM-2 提取蛋白质序列嵌入。
当前实验使用 esm2_t33_650M_UR50D，第33层，1280维表示。
"""

import numpy as np
import torch
from tqdm import tqdm


class ESM2Embedder:
    """ESM-2 序列嵌入提取器"""

    def __init__(self, model_name=None, device=None):
        """
        Args:
            model_name: ESM-2模型名称, 默认从config读取。可选:
                - esm2_t33_650M_UR50D (650M参数, 1280D, ~2.6GB显存)
            device: "cuda" / "cpu" / None(自动检测)
        """
        from config import ESM2_MODEL_NAME
        model_name = model_name or ESM2_MODEL_NAME
        if model_name != "esm2_t33_650M_UR50D":
            raise ValueError("The current experiment profile requires esm2_t33_650M_UR50D")

        self.model_name = model_name
        self.name = "M2_ESM2"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[ESM2] 加载模型 {model_name} -> {self.device}")
        self._load_model()

    def _load_model(self):
        import esm
        from pathlib import Path

        # 优先使用本地模型文件
        from config import MODELS_DIR
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(MODELS_DIR / "torch_hub"))

        local_path = Path(MODELS_DIR) / f"{self.model_name}.pt"
        if local_path.exists():
            print(f"[ESM2] 使用本地模型: {local_path}")
            model, alphabet = esm.pretrained.load_model_and_alphabet(str(local_path))
        else:
            print(f"[ESM2] 本地模型不存在 ({local_path}), 从 Hub 下载...")
            model, alphabet = getattr(esm.pretrained, self.model_name)()
        self.model = model.eval().to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        # 确定嵌入维度和层号
        self.repr_layer = 33
        self.dim = 1280
        print(f"[ESM2] 模型加载完成, 层={self.repr_layer}, 维度={self.dim}")

    def embed_single(self, seq):
        """对单条序列提取嵌入 (mean pooling) → (dim,)"""
        seq = seq.upper().strip()
        # ESM-2 最大长度1022
        if len(seq) > 1022:
            seq = seq[:1022]

        data = [("protein", seq)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])

        # Mean pooling: 排除 <cls> (位置0) 和 <eos> (最后)
        token_repr = results["representations"][self.repr_layer][0, 1:len(seq)+1, :]
        seq_repr = token_repr.mean(dim=0)

        return seq_repr.cpu().numpy()

    def embed_batch(self, sequences, batch_size=32):
        """批量提取嵌入"""
        all_embeddings = []
        n_batches = (len(sequences) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(sequences), batch_size),
                      total=n_batches, desc="    [ESM2] 嵌入", leave=False):
            batch_seqs = sequences[i:i+batch_size]
            # 截断过长序列
            batch_seqs = [s[:1022] if len(s) > 1022 else s for s in batch_seqs]

            data = [(f"seq_{j}", s.upper().strip()) for j, s in enumerate(batch_seqs)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.repr_layer])

            representations = results["representations"][self.repr_layer]

            for j, seq in enumerate(batch_seqs):
                token_repr = representations[j, 1:len(seq)+1, :]
                seq_repr = token_repr.mean(dim=0)
                all_embeddings.append(seq_repr.cpu().numpy())

        return np.array(all_embeddings)

    def embed_per_residue(self, seq):
        """提取 per-residue 嵌入 → (L, dim), 用于窗口切片"""
        seq = seq.upper().strip()
        
        # 分块处理长序列，避免被1022限制强制截断
        chunk_size = 1022
        all_token_reprs = []
        
        for start_idx in range(0, len(seq), chunk_size):
            end_idx = min(start_idx + chunk_size, len(seq))
            chunk_seq = seq[start_idx:end_idx]
            
            data = [("protein", chunk_seq)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.repr_layer])

            # 返回每个残基的嵌入 (排除 <cls> 和 <eos>)
            token_repr = results["representations"][self.repr_layer][0, 1:len(chunk_seq)+1, :]
            all_token_reprs.append(token_repr.cpu().numpy())
            
        return np.vstack(all_token_reprs)  # (L, dim)

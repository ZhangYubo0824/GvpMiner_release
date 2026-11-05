"""
M3: ProtBERT-BFD 蛋白质语言模型嵌入
====================================
使用 Rostlab 的 ProtBERT-BFD 提取蛋白质序列嵌入。
420M参数, 1024D输出, 半精度约0.8GB显存, 4GB显存可运行。
"""

import re
import numpy as np
import torch
import sys
from tqdm import tqdm


class ProtBERTEmbedder:
    """ProtBERT-BFD 序列嵌入提取器"""

    def __init__(self, model_name=None, device=None, use_half=True):
        if model_name is None:
            from config import MODELS_DIR
            model_name = str(MODELS_DIR / "prot_bert_bfd")

        self.model_name = model_name
        self.name = "M3_ProtBERT"
        self.dim = 1024
        self.use_half = use_half

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[ProtBERT] 加载模型 {model_name} -> {self.device}")
        self._load_model()

    def _load_model(self):
        """加载ProtBERT模型"""
        import os
        if not os.path.exists(self.model_name):
            raise FileNotFoundError(f"模型路径不存在: {self.model_name}")

        try:
            from transformers import BertTokenizer, BertModel
        except ImportError:
            print("错误: 请安装 transformers: pip install transformers")
            sys.exit(1)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(self.model_name)

        if self.use_half and self.device.type == "cuda":
            self.model = self.model.half()
            print("[ProtBERT] 使用半精度 (FP16), 显存占用约 0.8GB")

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[ProtBERT] 模型加载完成, 嵌入维度={self.dim}")

    def _prepare_sequence(self, seq):
        """预处理序列: 加空格 + 替换非标准氨基酸"""
        seq = seq.upper().strip()
        seq_spaced = " ".join(list(seq))
        seq_spaced = re.sub(r"[UZOB]", "X", seq_spaced)
        return seq_spaced

    def embed_single(self, seq):
        """对单条序列提取嵌入 (mean pooling) → (1024,)"""
        seq_prepared = self._prepare_sequence(seq)

        ids = self.tokenizer(seq_prepared, return_tensors="pt",
                             add_special_tokens=True, truncation=True, max_length=1024)
        ids = {k: v.to(self.device) for k, v in ids.items()}

        with torch.no_grad():
            output = self.model(**ids)

        # Mean pooling: 排除 [CLS] (位置0) 和 [SEP] (最后)
        token_repr = output.last_hidden_state[0, 1:-1, :]  # (L, 1024)
        seq_repr = token_repr.mean(dim=0)  # (1024,)

        return seq_repr.cpu().float().numpy()

    def embed_batch(self, sequences, batch_size=16):
        """批量提取嵌入"""
        all_embeddings = []
        n_batches = (len(sequences) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(sequences), batch_size),
                      total=n_batches, desc="    [ProtBERT] 嵌入", leave=False):
            batch_seqs = sequences[i:i+batch_size]
            batch_prepared = [self._prepare_sequence(s) for s in batch_seqs]

            ids = self.tokenizer(batch_prepared, return_tensors="pt",
                                 padding=True, truncation=True, max_length=1024,
                                 add_special_tokens=True)
            ids = {k: v.to(self.device) for k, v in ids.items()}

            with torch.no_grad():
                output = self.model(**ids)

            for j in range(len(batch_seqs)):
                # 找到非padding的有效长度
                attn_mask = ids["attention_mask"][j]
                valid_len = attn_mask.sum().item()
                # 排除 [CLS] 和 [SEP]
                token_repr = output.last_hidden_state[j, 1:valid_len-1, :]
                if token_repr.shape[0] == 0:
                    all_embeddings.append(np.zeros(self.dim))
                else:
                    seq_repr = token_repr.mean(dim=0)
                    all_embeddings.append(seq_repr.cpu().float().numpy())

        return np.array(all_embeddings)

    def embed_per_residue(self, seq):
        """提取 per-residue 嵌入 → (L, 1024)"""
        seq = seq.upper().strip()
        
        # 分块处理长序列，避免大于1024时被强制截断导致越界
        chunk_size = 1000  # 留足空间给特殊token
        all_token_reprs = []
        
        for start_idx in range(0, len(seq), chunk_size):
            end_idx = min(start_idx + chunk_size, len(seq))
            chunk_seq = seq[start_idx:end_idx]
            seq_prepared = self._prepare_sequence(chunk_seq)

            ids = self.tokenizer(seq_prepared, return_tensors="pt",
                                 add_special_tokens=True, truncation=True, max_length=1024)
            ids = {k: v.to(self.device) for k, v in ids.items()}

            with torch.no_grad():
                output = self.model(**ids)

            # 排除 [CLS] 和 [SEP] (1:-1)
            token_repr = output.last_hidden_state[0, 1:-1, :]  # (chunk_L, 1024)
            all_token_reprs.append(token_repr.cpu().float().numpy())
            
        return np.vstack(all_token_reprs)

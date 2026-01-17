"""
Setup script and requirements for URM GSM-8K implementation

Installation:
    pip install -r requirements.txt
    python setup_and_run.py
"""

# ==================== requirements.txt ====================
REQUIREMENTS = """
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
tqdm>=4.65.0
"""

# ==================== Quick Start Script ====================
QUICK_START = """
#!/usr/bin/env python3
'''
Quick start script for URM GSM-8K training

Usage:
    python quick_start.py --samples 100 --epochs 5
    python quick_start.py --samples 50 --epochs 3 --test-only
'''

import torch
import argparse
from urm_gsm8k import (
    UniversalReasoningModel,
    GSM8KDataset,
    train_urm,
    generate_answer
)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='URM GSM-8K Training')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of training samples (default: 100)')
    parser.add_argument('--test-samples', type=int, default=20,
                        help='Number of test samples (default: 20)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--d-model', type=int, default=256,
                        help='Model dimension (default: 256)')
    parser.add_argument('--n-loops', type=int, default=6,
                        help='Number of recurrent loops (default: 6)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run test generation (requires saved model)')
    parser.add_argument('--model-path', type=str, default='urm_gsm8k_best.pt',
                        help='Path to save/load model')
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    print("Creating model...")
    model = UniversalReasoningModel(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=4,
        d_ff=args.d_model * 4,
        n_layers=2,
        n_loops=args.n_loops,
        max_seq_len=512,
        dropout=0.1,
        tbptt_window=3
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Recurrent loops: {args.n_loops}")
    
    if args.test_only:
        # Load model and test
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)
        
        test_questions = [
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. How much more money does Betty need?",
            "A store sells pencils for $0.50 each. If John buys 12 pencils, how much does he pay?",
        ]
        
        print("\\n=== Testing Generation ===")
        for i, q in enumerate(test_questions, 1):
            print(f"\\n[{i}] Q: {q}")
            answer = generate_answer(model, tokenizer, q, device)
            print(f"    A: {answer}")
    else:
        # Train
        print("Loading datasets...")
        train_dataset = GSM8KDataset('train', tokenizer, num_samples=args.samples)
        test_dataset = GSM8KDataset('test', tokenizer, num_samples=args.test_samples)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Train
        print("\\nStarting training...")
        model = train_urm(
            model,
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=args.model_path
        )
        
        # Quick test
        print("\\n=== Quick Test ===")
        q = "A store has 100 apples. If they sell 30, how many are left?"
        print(f"Q: {q}")
        answer = generate_answer(model, tokenizer, q, device)
        print(f"A: {answer}")
    
    print("\\n=== Done ===")


if __name__ == '__main__':
    main()
"""

# ==================== README ====================
README = """
# URM (Universal Reasoning Model) for GSM-8K

シンプルなWSL環境向けのURM実装です。論文の核心的なアイデアを取り入れつつ、
公開されているコードには依存しない独自実装です。

## 主な特徴

### 論文のコア技術
1. **Recurrent Transformer Layers**: 同じレイヤーを複数回適用(ループ)
2. **ConvSwiGLU**: 局所的なトークン間相互作用のための畳み込み
3. **Truncated BPTT**: 勾配爆発を防ぐための切り詰めバックプロパゲーション

### シンプルな設計
- GPU 1枚で動作
- 小規模なサンプル数(100個程度)で学習可能
- 重いSudokuタスクは不要
- TRM/HRMとの比較は不要

## セットアップ

### 1. 環境構築
```bash
# 仮想環境作成(推奨)
python -m venv urm_env
source urm_env/bin/activate  # WSL/Linux

# 依存パッケージインストール
pip install torch transformers datasets tqdm
```

### 2. ファイル構成
```
urm_gsm8k/
├── urm_gsm8k.py         # メインモデル実装
├── quick_start.py       # 実行スクリプト
└── README.md            # このファイル
```

## 使い方

### 基本的な学習
```bash
# 100サンプルで5エポック学習
python quick_start.py --samples 100 --epochs 5

# より小規模なテスト(50サンプル、3エポック)
python quick_start.py --samples 50 --epochs 3
```

### カスタマイズオプション
```bash
# ループ数を変更(URMの重要パラメータ)
python quick_start.py --samples 100 --n-loops 8

# モデルサイズを変更
python quick_start.py --samples 100 --d-model 512

# バッチサイズを調整(GPU メモリに応じて)
python quick_start.py --samples 100 --batch-size 2
```

### 学習済みモデルでテスト
```bash
# 保存されたモデルで推論のみ
python quick_start.py --test-only --model-path urm_gsm8k_best.pt
```

## モデル構造

```
UniversalReasoningModel
├── Token Embedding
├── Positional Encoding
├── Recurrent Loops (n_loops=6)
│   └── For each loop:
│       ├── URMLayer 1
│       │   ├── Multi-Head Attention
│       │   └── ConvSwiGLU FFN
│       └── URMLayer 2
│           ├── Multi-Head Attention
│           └── ConvSwiGLU FFN
└── Output Projection
```

### ConvSwiGLU
```python
# 標準的なSwiGLU
H = SiLU(Gate) ⊙ Up

# ConvSwiGLU (局所的相互作用を追加)
H = Conv1D(SiLU(Gate) ⊙ Up)
```

## パラメータ説明

### モデルパラメータ
- `d_model` (256): モデルの次元数
- `n_heads` (4): アテンションヘッド数
- `d_ff` (1024): フィードフォワード層の次元数
- `n_layers` (2): レイヤー数
- `n_loops` (6): **重要** - 再帰的ループの回数
- `tbptt_window` (3): 勾配計算するループの窓サイズ

### 学習パラメータ
- `lr` (1e-4): 学習率
- `weight_decay` (0.1): 重み減衰
- `epochs` (5): エポック数
- `batch_size` (4): バッチサイズ

## メモリ使用量

| 設定 | GPU メモリ | 備考 |
|------|-----------|------|
| d_model=256, batch=4 | ~2-3GB | 推奨設定 |
| d_model=512, batch=2 | ~4-5GB | より大きなモデル |
| d_model=256, batch=8 | ~4-6GB | バッチサイズ増加 |

## トラブルシューティング

### CUDA Out of Memory
```bash
# バッチサイズを減らす
python quick_start.py --batch-size 2

# モデルサイズを小さくする
python quick_start.py --d-model 128
```

### 学習が遅い
```bash
# サンプル数を減らす
python quick_start.py --samples 50

# ループ数を減らす(精度は下がる可能性あり)
python quick_start.py --n-loops 4
```

## 論文との違い

### 実装済み
✅ Universal Transformer (recurrent loops)
✅ ConvSwiGLU
✅ Truncated BPTT
✅ RMSNorm
✅ Weight tying

### 簡略化した部分
❌ Adaptive Computation Time (ACT)
❌ ARC-AGI データセット
❌ Sudoku データセット
❌ 複数GPU並列化
❌ 大規模なデータ拡張

## 参考文献

```bibtex
@misc{gao2025universalreasoningmodel,
    title={Universal Reasoning Model}, 
    author={Zitian Gao and Lynx Chen and Yihao Xiao and He Xing and Ran Tao and Haoming Luo and Joey Zhou and Bryan Dai},
    year={2025},
    eprint={2512.14693},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2512.14693}
}
```

## ライセンス

このコードは教育目的のサンプル実装です。論文の手法を参考にしていますが、
公式実装 (https://github.com/UbiquantAI/URM) とは独立した実装です。
"""

# ==================== Save Files ====================

def setup():
    """Create all necessary files"""
    import os
    
    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS.strip())
    print("✓ Created requirements.txt")
    
    # Create quick_start.py
    with open('quick_start.py', 'w') as f:
        f.write(QUICK_START.strip())
    os.chmod('quick_start.py', 0o755)
    print("✓ Created quick_start.py")
    
    # Create README
    with open('README.md', 'w') as f:
        f.write(README.strip())
    print("✓ Created README.md")
    
    print("\n=== Setup Complete ===")
    print("Next steps:")
    print("1. pip install -r requirements.txt")
    print("2. python quick_start.py --samples 100 --epochs 5")


if __name__ == '__main__':
    setup()

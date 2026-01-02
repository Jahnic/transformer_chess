# Transformer From Scratch: Chess Edition

A minimal but complete transformer implementation for educational purposes.
Trained on chess games to generate valid move sequences.

**Goal:** Build deep intuition for how transformers work by implementing every component from scratch, with rich visualizations at each step.

## Quick Start (Mac Mini M4)

```bash
# 1. Clone/download this project
cd transformer_chess

# 2. Run setup (creates venv, installs dependencies)
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. Option A: Download real chess data from Lichess
cd data && bash download.sh && cd ..

# 4. Option B: Create synthetic data for quick testing
python src/data.py

# 5. Explore the data (start here!)
jupyter notebook notebooks/01_phase0_data_exploration.ipynb
```

## Project Structure

```
transformer_chess/
├── data/
│   ├── raw/              # Original PGN files
│   ├── processed/        # Clean training data (train.txt, val.txt, vocab.json)
│   └── download.sh       # Data acquisition script
├── src/
│   ├── data.py           # PGN parsing and preprocessing
│   ├── dataset.py        # PyTorch Dataset and DataLoader
│   ├── model.py          # Transformer architecture (built incrementally)
│   ├── train.py          # Training loop
│   └── visualize.py      # Attention and embedding visualizations
├── notebooks/            # Jupyter notebooks for each phase
│   └── 01_phase0_data_exploration.ipynb
├── checkpoints/          # Saved model weights
├── outputs/              # Generated samples
├── requirements.txt      # Python dependencies
├── setup.sh             # One-command setup
└── README.md
```

## Learning Phases

| Phase | Focus | Key Insight |
|-------|-------|-------------|
| **0** | Data | Understand what the model will see |
| **1** | Embeddings | Numbers in, vectors out |
| **2** | Single-head attention | The "where to look" mechanism |
| **3** | Multi-head attention | Different heads learn different patterns |
| **4** | Feed-forward | Where computation happens |
| **5** | Residuals + LayerNorm | Why deep networks train |
| **6** | Full stack | Putting it all together |
| **7** | Ablations | Break things to understand them |

## Data Format

Chess games are represented as space-separated moves:

```
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O
```

Character vocabulary (~26-30 tokens):
- **Files:** a-h (columns on the board)
- **Ranks:** 1-8 (rows on the board)  
- **Pieces:** K, Q, R, B, N (King, Queen, Rook, Bishop, Knight)
- **Special:** O (castling), x (capture), - (castling separator)
- **Structural:** space, newline

## Hardware Notes

On an M4 Pro Mac Mini:
- **MPS acceleration:** PyTorch automatically uses Metal for GPU acceleration
- **Expected training time:** Minutes to hours depending on model size
- **Sweet spot:** ~10-50M parameters, 4-8 layers

## Resources

- [Lichess Database](https://database.lichess.org/) - Free chess game downloads
- [python-chess](https://python-chess.readthedocs.io/) - Move validation
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper

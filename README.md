# BERT Fine-Tuning for News Classification

Fine-tuned Google's BERT base uncased model on the NYT dataset using an A100 GPU on Google Colab to categorize news articles into 3 imbalanced classes (business, sports, and politics) achieving a validation F1 score of 95.28% and test F1 score of 96.34%. Also implemented and compared Word2Vec, GloVe, and traditional vector embedding algorithms (Bag-of-Words, TF-IDF).

## 📊 Dataset

- **Source**: New York Times (NYT) articles
- **Classes**: Business, Sports, Politics (imbalanced distribution)
- **Total Samples**: 11,519
- **Split**: 80% train, 10% validation, 10% test (stratified)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: A100 or equivalent)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sharmeen-k/BERT_Fine_Tuning.git
cd BERT_Fine_Tuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU support (CUDA 11.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Setup

Place your `nyt.csv` file in the project root directory or update the path in the notebook:
```python
df = pd.read_csv('nyt.csv')  # Update this path as needed
```

## 📓 Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook Kazi_Sharmeen_CS235_HW3.ipynb
```

The notebook includes implementations for:
- **Part 1**: Bag-of-Words (binary, frequency, TF-IDF vectors)
- **Part 2**: Word2Vec and GloVe embeddings
- **Part 3**: BERT fine-tuning

## 📈 Results Summary

### Model Performance Ranking (Macro F1 Score)

| Rank | Model | Macro F1 Score |
|------|-------|----------------|
| 1 | TF-IDF | 0.9798 |
| 2 | Frequency Vector | 0.9724 |
| 3 | GloVe | 0.9706 |
| 4 | Binary Vector | 0.9690 |
| 5 | BERT Fine-tuned | 0.9634 |
| 6 | Word2Vec | 0.9594 |

### Key Findings

**Imbalanced Dataset Handling**: Initial experiments without stratification yielded poor macro F1 scores. Implementing stratified splitting significantly improved performance across underrepresented classes.

**TF-IDF vs Other BoW Methods**: TF-IDF outperformed both binary and frequency vectors (0.9798 vs 0.9690 and 0.9724), demonstrating the importance of inverse document frequency weighting for this task.

**GloVe vs Word2Vec**: GloVe embeddings (0.9706) outperformed Word2Vec (0.9594), likely due to GloVe being pre-trained on a much larger corpus (Wikipedia + Gigaword) compared to our limited training data.

**BERT Fine-tuning Observations**: 
- Training loss decreased across all 3 epochs
- Best performance achieved in epoch 2 (macro F1: 0.9528)
- Epoch 3 showed signs of overfitting (macro F1 dropped to 0.9392)
- Final test performance: 0.9634 macro F1

**Surprising Result**: Traditional TF-IDF outperformed fine-tuned BERT, possibly due to:
- Small dataset size (11,519 samples)
- Class imbalance issues
- Limited fine-tuning epochs
- BERT's tendency to overfit on smaller datasets

## 🔧 Model Architecture

### BERT Configuration
- **Base Model**: `bert-base-uncased`
- **Classifier**: Linear layer (768 → 3 classes)
- **Max Sequence Length**: 64 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Epochs**: 3
- **Hardware**: Google Colab A100 GPU

## 📝 Evaluation Metrics

- **Macro F1 Score**: Primary metric (unweighted average, prioritizes underrepresented classes)
- **Micro F1 Score**: Overall accuracy metric
- **Accuracy**: Standard classification accuracy

## 🔮 Future Improvements

- [ ] Implement mixed precision training (fp16/bf16) for A100 optimization
- [ ] Increase batch size to leverage A100 memory (40GB/80GB)
- [ ] Add learning rate warmup scheduler
- [ ] Implement early stopping to prevent overfitting
- [ ] Try larger BERT variants (bert-large, RoBERTa)
- [ ] Experiment with class weighting for imbalanced data
- [ ] Add gradient accumulation for effective larger batch sizes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google's BERT team for the pre-trained models
- Transformers library by Hugging Face
- NYT for the dataset
- Google Colab for providing A100 GPU access

## 📧 Contact

Kazi Sharmeen - sharmeenk666@gmail.com

---

**Note**: This was developed as part of CS235 coursework.

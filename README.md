# 🎓 AI-Powered MCQ Generator

> Fine-tuned LLaMA-3.2-1B model for generating high-quality multiple-choice questions on AI/ML topics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 Overview

This project fine-tunes Meta's LLaMA-3.2-1B-Instruct model to generate professional-quality multiple-choice questions (MCQs) for AI/ML topics. Using parameter-efficient fine-tuning (PEFT) with LoRA, the model achieves **85.4% overall quality score** while training on consumer hardware.

## ✨ Key Features

- ✅ **High-Quality Generation:** 85.4% average quality score
- ✅ **Domain-Specific:** Trained on 3,000+ curated AI/ML MCQs  
- ✅ **Parameter-Efficient:** Uses LoRA - only 0.9% trainable parameters
- ✅ **Production-Ready:** Gradio web interface with batch generation
- ✅ **Fast Training:** 47 minutes on T4 GPU
- ✅ **Comprehensive Evaluation:** 4-metric framework (structure, completeness, diversity, clarity)

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Overall Quality** | 85.4% |
| Structural Validity | 100% |
| Completeness | 100% |
| Option Diversity | 41.4% |
| Answer Clarity | 100% |

## 🚀 Quick Start

### Installation

\`\`\`bash
# Clone repository
git clone https://github.com/SaadImam2000/mcq-generator.git
cd mcq-generator

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Usage

\`\`\`python
from src.model import MCQGenerator

# Initialize generator
generator = MCQGenerator(
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    adapter_path="./models/adapter"
)

# Generate MCQ
mcq = generator.generate("What is deep learning?")
print(mcq)
\`\`\`

### Web Interface

\`\`\`bash
# Launch Gradio interface
python app.py

# Access at http://localhost:7860
\`\`\`

## 📁 Project Structure

\`\`\`
mcq-generator/
├── README.md
├── requirements.txt
├── models/
│   └── adapter/             # Trained LoRA adapter
├── results/
│   ├── metrics/             # Evaluation results
│   ├── examples/            # Showcase MCQs
│   └── visualizations/      # Training curves
├── mcq_generator.py
\`\`\`

## 🎓 Model Architecture

- **Base Model:** LLaMA-3.2-1B-Instruct
- **Fine-tuning:** LoRA (rank=16, alpha=32)
- **Trainable Params:** 11.27M (0.9% of base model)
- **Training Time:** 47 minutes on T4 GPU
- **Dataset:** 3,000 curated AI/ML MCQs

## 📈 Training Details

\`\`\`yaml
Training Configuration:
  batch_size: 4
  gradient_accumulation: 2
  learning_rate: 3e-4
  epochs: 5
  warmup_steps: 200
  optimizer: AdamW

Final Metrics:
  train_loss: 0.0065
  val_loss: 0.0066
  quality_score: 85.4%
\`\`\`

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

**Note:** LLaMA model has separate license terms from Meta.

## 📞 Contact

- **GitHub:** [@SaadImam2000](https://github.com/SaadImam2000)
- **Email:** saad.imam08@gmail.com

---

<div align="center">
Made for educators and learners worldwide
</div>
\`\`\`


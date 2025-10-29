# 🎓 AI-Powered MCQ Generator

> Fine-tuned LLaMA-3.2-1B model for generating high-quality multiple-choice questions on AI/ML topics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**[🚀 Live Demo](https://67d754a8aa74e2b5fb.gradio.live)** | **[📖 Documentation](docs/)** | **[🤗 Model](models/)**

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
git clone https://github.com/YOUR_USERNAME/mcq-generator.git
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
├── app.py                    # Gradio web interface
├── src/
│   ├── model.py             # Model loading and inference
│   ├── trainer.py           # Training pipeline
│   ├── evaluator.py         # Evaluation framework
│   └── data_utils.py        # Dataset utilities
├── models/
│   └── adapter/             # Trained LoRA adapter
├── results/
│   ├── metrics/             # Evaluation results
│   ├── examples/            # Showcase MCQs
│   └── visualizations/      # Training curves
└── docs/
    └── TRAINING.md          # Training guide
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

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

**Note:** LLaMA model has separate license terms from Meta.

## 📞 Contact

- **GitHub:** [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email:** your.email@example.com
- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/mcq-generator/issues)

---

<div align="center">
Made with ❤️ for educators and learners worldwide
</div>
\`\`\`

#### 📄 `LICENSE`
```python
%%writefile /content/mcq-generator/LICENSE
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#### 📄 `app.py`

Save your Week 2 Gradio code as `app.py`:
```python
%%writefile /content/mcq-generator/app.py
# Copy the entire Week 2 Demo code here
# (The code from week2-demo-docs artifact)
```

### Step 3: Create Source Files

#### 📄 `src/model.py`
```python
%%writefile /content/mcq-generator/src/model.py
"""
Model loading and inference for MCQ generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

class MCQGenerator:
    """Production-ready MCQ generator."""
    
    def __init__(self, base_model_name, adapter_path):
        """Initialize the generator with base model and adapter."""
        print("🔄 Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        print("✅ Model loaded!")
    
    def generate(self, question, temperature=0.7, top_p=0.9, max_tokens=200):
        """Generate MCQ from a question."""
        prompt = (
            f"Generate a multiple-choice question with exactly 4 options (A-D). "
            f"Mark the correct answer.\n\n"
            f"Question: {question}\nA) "
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = text.split("A) ")[-1] if "A) " in text else text.split(prompt)[-1]
        
        return f"Question: {question}\nA) {generated}"
```

### Step 4: Download from Colab
```python
# Zip everything
!cd /content && zip -r mcq-generator.zip mcq-generator/ -x "*/.*" "*.ipynb_checkpoints*"

# Download
from google.colab import files
files.download('/content/mcq-generator.zip')
```

---

## 📤 Upload to GitHub

### Method 1: GitHub Web Interface (Easiest)

1. **Create Repository**
   - Go to https://github.com/new
   - Name: `mcq-generator`
   - Description: "AI-powered MCQ generator using fine-tuned LLaMA-3.2-1B"
   - Make it **Public**
   - ✅ Add README (skip, you have one)
   - ✅ Add .gitignore (skip, you have one)
   - ✅ Choose MIT License (skip, you have one)
   - Click "Create repository"

2. **Upload Files**
   - Extract your zip file locally
   - On GitHub repo page, click "uploading an existing file"
   - Drag and drop ALL files/folders
   - Commit message: "Initial commit: MCQ generator with trained model"
   - Click "Commit changes"

### Method 2: Git Command Line (Recommended)
```bash
# Extract your zip file
unzip mcq-generator.zip
cd mcq-generator

# Initialize git
git init
git add .
git commit -m "Initial commit: MCQ generator with trained model"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/mcq-generator.git

# Push
git branch -M main
git push -u origin main
```

### Method 3: GitHub Desktop (User-Friendly)

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Extract your zip
3. In GitHub Desktop: File → Add Local Repository
4. Choose your mcq-generator folder
5. Commit all files
6. Click "Publish repository"

---

## 🔧 Important: Model Files

### Option A: Don't Upload Model (Recommended)

The trained model adapter is ~50MB. Instead:

1. **Upload to Hugging Face**:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/content/mcq_model_aiml_v2",
    repo_id="YOUR_USERNAME/llama-mcq-adapter",
    repo_type="model",
)
```

2. **Update README** with download instructions:
```markdown
## 📥 Download Trained Model

\`\`\`bash
# Download from Hugging Face
huggingface-cli download YOUR_USERNAME/llama-mcq-adapter --local-dir ./models/adapter
\`\`\`
```

### Option B: Use Git LFS (If You Want Model in Repo)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "models/adapter/*.bin"
git lfs track "models/adapter/*.safetensors"

git add .gitattributes
git add models/
git commit -m "Add model with Git LFS"
git push
```

---

## ✅ Post-Upload Checklist

After uploading to GitHub:

- [ ] Repository is public
- [ ] README displays correctly
- [ ] All folders visible
- [ ] Links work (update placeholders)
- [ ] License file present
- [ ] requirements.txt correct
- [ ] .gitignore working (no __pycache__, etc.)
- [ ] Add topics/tags (machine-learning, llm, gradio, education)
- [ ] Create GitHub Release (v1.0.0)

---

## 🎨 Make It Look Professional

### 1. Add Repository Topics

On GitHub, click ⚙️ next to "About" and add:
- `machine-learning`
- `nlp`
- `llm`
- `llama`
- `education`
- `gradio`
- `lora`
- `peft`
- `question-generation`

### 2. Update About Section

- Description: "AI-powered MCQ generator using fine-tuned LLaMA-3.2-1B with LoRA. 85% quality score."
- Website: Your live demo URL
- Topics: (added above)

### 3. Add a Banner Image

Create and add to README:
```markdown
![MCQ Generator Banner](assets/banner.png)
```

### 4. Create Release

- Go to "Releases" → "Create new release"
- Tag: `v1.0.0`
- Title: "Initial Release: MCQ Generator v1.0"
- Description:
```markdown
## 🎉 First Release

### Features
- ✅ Fine-tuned LLaMA-3.2-1B model
- ✅ 85.4% quality score
- ✅ Gradio web interface
- ✅ Batch generation support

### Performance
- Training time: 47 minutes
- Quality score: 85.4%
- Structural validity: 100%

### Downloads
- Model adapter: [Hugging Face](link)
- Demo: [Live](link)
```

---

...

git clone https://github.com/SaadImam2000/mcq-generator.git

...

- **GitHub:** [@SaadImam2000](https://github.com/SaadImam2000)
- **Email:** Saad.imam08@gmail.com
```

---

## 🚀 Final Steps

### 1. Share Your Repository

LinkedIn post:

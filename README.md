# AI Republic Project 1 (Fine-Tuned Gemma Model for Mental Health Conversations)

![Project Image](https://raw.githubusercontent.com/MFlores01/airepublic_project1_submission/refs/heads/main/MH_image.webp)


A chatbot trained on gemma2b-it fine-tuned on a dataset containing questions and answers regarding mental health conversations.

# Model Summary

This model is a fine-tuned version of gemma-2b-it for mental health counseling conversations. It was fine-tuned on the "Amod/Mental Health Counseling Conversations dataset", which contains dialogues related to mental health counseling. 
* Dataset:  [here](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
* Fine-Tuned Model: [here](https://huggingface.co/GuelGaMesh01/gemma-2b-it-finetuned-mental-health-qa)
* Base Model: [here](https://huggingface.co/google/gemma-2-2b-it)

# Repository Info
This repository contains two Jupyter notebooks demonstrating how to fine-tune a large language model (LLM) using the Hugging Face Transformers library, followed by a notebook that performs inference with the fine-tuned model. This project is part of the activities for Week 2 of the AI Republic's AI Bootcamp.

# Notebooks
### 1. Fine-tuned LLM Model (Gemma)
* Notebook Name: `3_Finetuninng_Gemma_2b_it_MentalHealth.ipynb`
* Description: This notebook shows you through the procedure of fine-tuning a pre-trained LLM on a custom dataset. It includes:
  * Loading a pre-trained model and tokenizer
  * Preparing the dataset for training
  * Fine-tuning the model
  * Uploading the model in HuggingFace
  * Evaluation Metrics

### 2. Inferenced Fine-tuned Model
* Notebook Name: `4_Inferencing_Finetuned_Model.ipynb`
* Description: In this notebook, you will use the fine-tuned model to make responses. It includes:
  * Loading the fine-tuned model and tokenizer
  * Preparing input data for inference
  * Running inference and displaying results

# Use Cases

### Direct Use
This model is fine-tuned for generating responses related to mental health counseling tasks. It can be used for providing suggestions, conversation starters, or follow-ups in mental health scenarios.

### Downstream Use 
This model can be adapted for use in more specific counseling-related tasks, or in applications where generating mental health-related dialogue is necessary.

### Out-of-Scope Use
The model is not intended to replace professional counseling. It should not be used for real-time crisis management or any situation requiring direct human intervention. Use in highly critical or urgent care situations is out of scope.

# Bias, Risks, and Limitations
The model was trained on mental health-related dialogues, but it may still generate biased or inappropriate responses. Users should exercise caution when interpreting or acting on the model's outputs, particularly in sensitive scenarios.

# Recommendations
The model should not be used as a replacement for professional mental health practitioners. Users should carefully evaluate generated responses in the context of their use case.

# How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("GuelGaMesh01/gemma-2b-it-finetuned-mental-health-qa") 
model = AutoModelForCausalLM.from_pretrained("GuelGaMesh01/gemma-2b-it-finetuned-mental-health-qa")
```

### Example inference
```python
inputs = tokenizer("How can I manage anxiety better?", return_tensors="pt") outputs = model.generate(**inputs, max_length=200) response = tokenizer.decode(outputs[0], skip_special_tokens=True) print(response)
```

# Training Details
### Training Data
The model was trained on the Amod/Mental Health Counseling Conversations dataset, which consists of mental health dialogues focused on counseling situations.

### Training Procedure
The model was fine-tuned using LoRA (Low-Rank Adaptation) with the following hyperparameters:

Batch Size: 1 Gradient Accumulation Steps: 4 Learning Rate: 2e-4 Epochs: 3 Max Sequence Length: 2000 tokens Optimizer: paged_adamw_8bit

### Training Hyperparameters
Training Time: Approximately 30 minutes for 100 steps.
Checkpoint Size: The model checkpoints are approximately 15 GB.

# Evaluation
### Testing Data
The model was evaluated using a split from the training data, specifically a 10% test split of the original training dataset.

### Metrics
The following metrics were used during the training and evaluation process:

Training Loss: The training loss was tracked during training to monitor how well the model was learning from the data. It decreased throughout the epochs.

Semantic Similarity: Semantic similarity was employed as the primary metric to assess the model’s ability to generate contextually relevant and meaningful responses. Since the dataset involves conversational context, particularly in the sensitive area of mental health counseling, it was crucial to evaluate how well the model understands and retains the intent and meaning behind the input rather than merely focusing on fluency or token-level prediction. 

Perplexity: Perplexity was used as a metric to evaluate the model's ability to generate coherent and fluent text responses. The model was evaluated on a subset of the test data, and both non-finetuned and finetuned perplexities were compared.

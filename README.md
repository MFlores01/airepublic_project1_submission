# AI Republic Project 1 (HuggingFace Gemma Model)

A chatbot trained on gemma2b-it fine-tuned on a dataset containing questions and answers regarding mental health conversations.

## Model Summary

This model is a fine-tuned version of gemma-2b-it for mental health counseling conversations. It was fine-tuned on the "Amod/Mental Health Counseling Conversations dataset", which contains dialogues related to mental health counseling. Access the dataset through this link: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations


## Use Cases

### Direct Use
This model is fine-tuned for generating responses related to mental health counseling tasks. It can be used for providing suggestions, conversation starters, or follow-ups in mental health scenarios.

### Downstream Use 
This model can be adapted for use in more specific counseling-related tasks, or in applications where generating mental health-related dialogue is necessary.

## Out-of-Scope Use
The model is not intended to replace professional counseling. It should not be used for real-time crisis management or any situation requiring direct human intervention. Use in highly critical or urgent care situations is out of scope.

## Bias, Risks, and Limitations
The model was trained on mental health-related dialogues, but it may still generate biased or inappropriate responses. Users should exercise caution when interpreting or acting on the model's outputs, particularly in sensitive scenarios.

## Recommendations
The model should not be used as a replacement for professional mental health practitioners. Users should carefully evaluate generated responses in the context of their use case.

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("GuelGaMesh01/gemma-2b-it-finetuned-mental-health-qa") 
model = AutoModelForCausalLM.from_pretrained("GuelGaMesh01/gemma-2b-it-finetuned-mental-health-qa")
```

# Example inference
```python
inputs = tokenizer("How can I manage anxiety better?", return_tensors="pt") outputs = model.generate(**inputs, max_length=200) response = tokenizer.decode(outputs[0], skip_special_tokens=True) print(response)
```

# Training Details
# Training Data
The model was trained on the Amod/Mental Health Counseling Conversations dataset, which consists of mental health dialogues focused on counseling situations.

# Training Procedure
The model was fine-tuned using LoRA (Low-Rank Adaptation) with the following hyperparameters:

Batch Size: 1 Gradient Accumulation Steps: 4 Learning Rate: 2e-4 Epochs: 3 Max Sequence Length: 2500 tokens Optimizer: paged_adamw_8bit

# Training Hyperparameters
Training Time: Approximately 30 minutes for 150 steps with fp16 mixed precision.
Checkpoint Size: The model checkpoints are approximately 15 GB.

# Evaluation
# Testing Data
The model was evaluated using a split from the training data, specifically a 10% test split of the original training dataset.

# Metrics
The following metrics were used during the training and evaluation process:

Training Loss: The training loss was tracked during training to monitor how well the model was learning from the data. It decreased throughout the epochs.
Perplexity: Perplexity was used as a metric to evaluate the model's ability to generate coherent and fluent text responses. The model was evaluated on a subset of the test data, and both non-finetuned and finetuned perplexities were compared.

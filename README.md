# SWE-Smith-DataScience

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/1ddad2ed-1234-46d3-aaac-95394fda7e1f"
    alt="ChatGPT Image 15 may 2025, 12_46_16 p m"
    width="200"
  />
</p>

This project fine-tunes an open-source AI model to specialize in Data Science tasks using a portion of the SWE-Smith dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)


## Overview

SWE-Smith-DataScience provides a modular pipeline to fine-tune large language models on specialized data science problems extracted from the SWE-Smith dataset and evaluate them via the SWE-agent framework.

## Features

- **Model Fine-Tuning**: Leverage `finetune.py` to train on data subsets.  
- **Automated Evaluation**: Use the `SWE-agent` module to benchmark performance.  
- **Reproducible Workflows**: Modular scripts for data processing and training.

## Prerequisites

- Python 3.8 or higher  
- Git  
- (Optional) [Modal](https://modal.com/) CLI for scalable remote training  
- Required Python packages:
  ```bash
  pip install torch transformers datasets modal

## Installation
```bash
git clone https://github.com/HugoGoHe/SWE-Smith-DataScience.git
cd SWE-Smith-DataScience
```

## Project Structure
```text
SWE-Smith-DataScience/
├── SWE-agent/           # Evaluation agent and benchmarking scripts
├── finetune.py          # Fine-tuning script for the model
├── requirements.txt     # Python dependencies (optional)
├── LICENSE              # MIT License
└── README.md            # Project documentation
```

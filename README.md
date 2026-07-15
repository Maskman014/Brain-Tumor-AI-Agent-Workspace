# 🧠 Autonomous Brain Tumor Classification & Analysis Agent

## 📋 Project Overview
This repository contains a full-stack, tool-calling AI Agent pipeline designed for computer vision-based clinical decision support. The architecture integrates a classical machine learning core with a modern Large Language Model (LLM) reasoning loop, deployed via an interactive user interface.

## ⚙️ System Architecture
The application runs inside a Google Colab notebook instance utilizing a three-stage execution layer:
1. **Machine Learning Tool Engine:** A scikit-learn `RandomForestClassifier` trained on flattened image matrices extracted from axial/sagittal brain MRI scans (`yes`/`no` datasets).
2. **Autonomous Reasoning Loop:** An advanced generative AI connection utilizing `gemini-2.5-flash` configured with custom native system instructions to independently execute functional tool calls.
3. **Unified Interface & Ledger:** A reactive `Gradio` block UI featuring a real-time system output narrative terminal and a stateful data tracking ledger matrix.

## 🚀 Deployment Instructions
1. Open the `.ipynb` notebook file inside Google Colab.
2. Supply a valid Google AI Studio API key inside the initialization block.
3. Navigate to **Runtime > Run All** (`Ctrl + F9`) to spin up the local server environments.
4. Access the public `.gradio.live` link generated in the final cell execution log to interact with the runtime workspace.

## 🛠️ Technology Stack
* **Language:** Python 3.12
* **Core Frameworks:** Google GenAI SDK, Gradio UI Engine
* **Machine Learning & Vision:** Scikit-Learn, OpenCV (cv2), NumPy, Pandas

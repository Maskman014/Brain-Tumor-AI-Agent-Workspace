# 🧠 Autonomous Brain Tumor Classification & Analysis Agent

## 📋 Project Overview
This repository contains a full-stack, tool-calling AI Agent pipeline designed for computer vision-based clinical decision support. The architecture integrates a classical machine learning core with a modern Large Language Model (LLM) reasoning loop, deployed via an interactive user interface.

The system trains its primary machine learning vision tool on actual brain MRI scan structures, maps those evaluation matrices to an LLM runtime environment, and produces natural language diagnostic reports alongside an active database ledger.

---

## ⚙️ System Architecture & Workflow
The application executes within a structured three-tier execution layer inside Google Colab:

* **Machine Learning Tool Engine:** A scikit-learn `RandomForestClassifier` trained on flattened image matrices extracted from axial/sagittal brain MRI scans (`yes`/`no` datasets).
* **Autonomous Reasoning Loop:** An advanced generative AI connection utilizing `gemini-2.5-flash` configured with custom native system instructions to independently execute functional tool calls.
* **Unified Interface & Ledger:** A reactive `Gradio` block UI featuring a real-time system output narrative terminal and a stateful data tracking ledger matrix.

---

## 📥 Dataset Source
The core machine learning engine uses the public **Brain MRI Images for Brain Tumor Detection** dataset hosted on Kaggle.

* **Dataset URL:** [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* **Structure:** The dataset contains folders split into structural classes:
  * `yes/`: MRI scans indicating clear presence of space-occupying tumor masses.
  * `no/`: Control group MRI scans containing structural tissue within normal operational boundaries.

---

## 🚀 Deployment & Execution Instructions
Follow these steps to run the workspace environment locally or via cloud runtimes:

### 1. Prerequisites & API Configuration
* Obtain a valid Gemini API key from the [Google AI Studio Dashboard](https://aistudio.google.com/).
* Ensure your Kaggle account is active if you intend to download the raw archive programmatically via the Kaggle CLI API using your `kaggle.json` credentials.

### 2. Notebook Execution Step
* Upload the `.ipynb` notebook file inside a fresh Google Colab environment.
* In the Colab file tree explorer sidebar on the left, ensure your extracted `yes` and `no` image directories match the paths specified in your image processing loop (`/content/yes` and `/content/no`).
* Navigate to the top menu bar and select **Runtime > Run All** (`Ctrl + F9`).
* When prompted by the interactive terminal box, paste your Google AI Studio API Key securely into the masked environment input line.

### 3. Application Verification
* Once cell execution down to the interface layer finishes, click the generated public deployment share link (`https://xxxxxxxxxxxxxxxx.gradio.live`).
* Input a sample patient ID string, select your imaging modality sequence, upload an image matrix from the dataset collection, and execute the core diagnostics to populate the narrative report and transaction logs.

---

## 🛠️ Technology Stack
* **Language Environment:** Python 3.12
* **Core Frameworks:** Google GenAI SDK, Gradio UI Engine
* **Machine Learning & Vision:** Scikit-Learn, OpenCV (cv2), NumPy, Pandas
* **Security Layer:** Native Python `getpass` runtime variable masking

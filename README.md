# 🤖 AutoData Agent – Autonomous Hugging Face AI for Data Cleaning & Modeling

<div align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="120"/>
  <h3>An autonomous data science agent that cleans, explores and models your data intelligently 🧠</h3>
  <p>Built with <b>Hugging Face SmolAgents</b>, <b>Ollama</b>, and <b>Scikit-Learn</b></p>
</div>

---

## 🧠 Project Overview

**AutoData Agent** is a modular and autonomous data assistant designed to:

* 🧹 **Inspect** raw datasets (missing values, data types, anomalies)
* 🧼 **Clean** and preprocess data (imputation, encoding, scaling)
* 📊 **Visualize** key statistical properties (distributions, correlations, boxplots)
* 🤖 **Train machine learning models** automatically (classification or regression)
* ⚙️ **Optimize** the model using GridSearchCV to achieve best performance

This project demonstrates how a **Hugging Face agent** can orchestrate an end-to-end **Data Science pipeline**,  making smart decisions and reasoning about the dataset structure.

---

## 🧩 Architecture

```bash
datacleaner-agent/
├── app.py                 # Main entry point
├── agent_logic.py         # Builds the Hugging Face Agent (model + tools)
├── tools/
│   ├── inspect.py         # InspectTool: automatic EDA (plots + summary)
│   ├── cleaning.py        # CleaningTool: data cleaning & encoding
│   └── train.py           # TrainTool: automatic ML training + evaluation
├── test_tools.py          # Local tests for each tool
├── requirements.txt
└── README.md
```

---

## 🚀 How It Works

### 🔹 Step 1 — **InspectTool**

Performs an **Exploratory Data Analysis (EDA)**:

* Displays dataset info (shape, dtypes, missing values)
* Generates **histograms**, **boxplots**, and **correlation heatmaps**
* Detects data imbalances and null distributions

**Example output:**

* `df.info()`, `df.describe()`
* Automatic visualizations for numeric variables
* Summary of missing data

---

### 🔹 Step 2 — **CleaningTool**

Cleans and prepares the dataset:

* Removes duplicates
* Handles missing values (median or mode)
* Encodes categorical variables (LabelEncoder or OneHotEncoder)
* Scales numerical columns (StandardScaler)
* Detects and drops low-variance features

**Goal:** produce a dataset ready for model training.

---

### 🔹 Step 3 — **TrainTool**

Automatically trains a model based on target variable type:

* Detects whether it’s **classification** or **regression**
* Chooses the appropriate **RandomForest** model
* Runs **GridSearchCV** to optimize hyperparameters
* Splits data into train/test (80/20)
* Displays key metrics:

  * Classification: `Accuracy`, `Precision`, `Recall`, `F1`
  * Regression: `RMSE`, `R²`

---

## ⚙️ Installation

### Prerequisites

* Python ≥ 3.11
* Virtual environment recommended

```bash
git clone https://github.com/MeidiLprog/datacleaner-agent.git
cd datacleaner-agent
pip install -r requirements.txt
```

---

## 🧭 Usage

### ▶️ Run the agent

```bash
python app.py
```

The agent will:

1. Load the Titanic dataset (default)
2. Inspect and clean it automatically
3. Train a predictive model on `Survived`
4. Output key metrics and model summary

**Expected Output:**

```
Dataset successfully loaded ! (891, 12)
Agent ready !
Inspecting dataset...
Cleaning done...
GridSearch training...
Accuracy: 0.84
F1 Score: 0.81
```

---

## ☁️ Supported Execution Modes

### 🟡 Hugging Face Cloud (Recommended)

Runs the reasoning model via Hugging Face Inference API.

Set your token:

```bash
export HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

Then in `agent_logic.py`:

```python
model = LiteLLMModel(model_id="huggingface/mistralai/Mistral-7B-Instruct-v0.2")
```

### 🔵 Local (Offline) – Ollama

If you prefer running locally:

1. Install [Ollama](https://ollama.com)
2. Pull the model:

   ```bash
   ollama pull qwen2:1.5b
   ```
3. Replace model in `agent_logic.py`:

   ```python
   model = LiteLLMModel(model_id="ollama/qwen2:1.5b")
   ```

---

## 🧰 Technologies Used

| Stack                      | Purpose                            |
| -------------------------- | ---------------------------------- |
| 🤗 Hugging Face SmolAgents | Agent orchestration                |
| 🔮 LiteLLM / HfApiModel    | LLM reasoning                      |
| 🧹 Pandas / Numpy          | Data wrangling                     |
| 📊 Matplotlib / Seaborn    | Data visualization                 |
| ⚙️ Scikit-Learn            | Model training & GridSearch        |
| 💻 Ollama                  | Local LLM inference (offline mode) |

---

## 💡 Example Screenshots

| Visualization                    | Description                          |
| -------------------------------- | ------------------------------------ |
| ![EDA](assets/eda.png)           | Automatic data histograms & boxplots |
| ![Heatmap](assets/heatmap.png)   | Correlation matrix                   |
| ![Training](assets/training.png) | Model training output                |

---

## 🧑‍💻 Author

**Lefki Meidi**
🎓 Data Science & Machine Learning Engineer
💬 [LinkedIn](https://www.linkedin.com/in/lefkimeidi) • [GitHub](https://github.com/MeidiLprog) • [HuggingFace](https://huggingface.co/Meidilefki)

---

## 🌟 Project Highlights

* Built **entirely from scratch** in less than 24h
* Modular architecture (plug & play tools)
* Hugging Face AI agent integrated locally **and** via cloud API
* Fully autonomous workflow: from raw data → cleaned dataset → trained model
* Ideal for data preprocessing automation or teaching agent reasoning

---

## ❤️ Acknowledgements

Special thanks to **Hugging Face** for the SmolAgents framework, and the open-source community for making AI accessible.

> “Why spend hours cleaning data when your agent can do it for you?”

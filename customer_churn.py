{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "private_outputs": true,
      "provenance": [],
      "authorship_tag": "ABX9TyNAgKMlMM/Z4Y2wHV5R28sk",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Jainrani/Customer-Churn-Prediction/blob/main/app_py.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install matplotlib-venn"
      ],
      "metadata": {
        "id": "LlQFMT7kPmLK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!apt-get -qq install -y libfluidsynth1"
      ],
      "metadata": {
        "id": "tycuhPecPrYm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# https://pypi.python.org/pypi/libarchive\n",
        "!apt-get -qq install -y libarchive-dev && pip install -U libarchive\n",
        "import libarchive"
      ],
      "metadata": {
        "id": "YHYSuZ4DPylu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# https://pypi.python.org/pypi/pydot\n",
        "!apt-get -qq install -y graphviz && pip install pydot\n",
        "import pydot"
      ],
      "metadata": {
        "id": "DurVSpPuP4mv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install cartopy\n",
        "import cartopy"
      ],
      "metadata": {
        "id": "EewC9_qQP-sn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "atIMJjK2PD4J"
      },
      "outputs": [],
      "source": [
        "# app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pickle\n",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "# ----------------------------\n",
        "# Title\n",
        "# ----------------------------\n",
        "st.title(\"📉 Customer Churn Prediction App\")\n",
        "st.write(\"Predict whether a customer will churn based on their details.\")\n",
        "\n",
        "# ----------------------------\n",
        "# Load Dataset\n",
        "# ----------------------------\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    df = pd.read_csv(\"/content/WA_Fn-UseC_-Telco-Customer-Churn.csv\")\n",
        "    return df\n",
        "\n",
        "data = load_data()\n",
        "st.write(\"Sample Dataset:\")\n",
        "st.dataframe(data.head())\n",
        "\n",
        "# ----------------------------\n",
        "# Preprocessing\n",
        "# ----------------------------\n",
        "@st.cache_data\n",
        "def preprocess_data(df):\n",
        "    df = df.copy()\n",
        "    # Example preprocessing (update according to your dataset)\n",
        "    df.fillna(0, inplace=True)\n",
        "\n",
        "    # Encode categorical columns\n",
        "    label_cols = df.select_dtypes(include=['object']).columns\n",
        "    le_dict = {}\n",
        "    for col in label_cols:\n",
        "        le = LabelEncoder()\n",
        "        df[col] = le.fit_transform(df[col])\n",
        "        le_dict[col] = le\n",
        "    return df, le_dict\n",
        "\n",
        "processed_data, le_dict = preprocess_data(data)\n",
        "\n",
        "# ----------------------------\n",
        "# Train Model (Random Forest)\n",
        "# ----------------------------\n",
        "@st.cache_resource\n",
        "def train_model(df):\n",
        "    X = df.drop(\"Churn\", axis=1)\n",
        "    y = df[\"Churn\"]\n",
        "\n",
        "    scaler = StandardScaler()\n",
        "    X_scaled = scaler.fit_transform(X)\n",
        "\n",
        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "    model.fit(X_scaled, y)\n",
        "\n",
        "    return model, scaler, X.columns\n",
        "\n",
        "model, scaler, feature_cols = train_model(processed_data)\n",
        "\n",
        "# ----------------------------\n",
        "# User Input for Prediction\n",
        "# ----------------------------\n",
        "st.sidebar.header(\"Enter Customer Details\")\n",
        "\n",
        "user_input = {}\n",
        "for col in feature_cols:\n",
        "    if col in le_dict:  # Categorical column\n",
        "        options = list(le_dict[col].classes_)\n",
        "        user_input[col] = st.sidebar.selectbox(col, options)\n",
        "    else:  # Numerical column\n",
        "        min_val = int(data[col].min())\n",
        "        max_val = int(data[col].max())\n",
        "        mean_val = int(data[col].mean())\n",
        "        user_input[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)\n",
        "\n",
        "# ----------------------------\n",
        "# Prediction\n",
        "# ----------------------------\n",
        "if st.sidebar.button(\"Predict Churn\"):\n",
        "    # Convert input to DataFrame\n",
        "    input_df = pd.DataFrame([user_input])\n",
        "\n",
        "    # Encode categorical inputs\n",
        "    for col in le_dict:\n",
        "        input_df[col] = le_dict[col].transform(input_df[col])\n",
        "\n",
        "    # Scale input\n",
        "    input_scaled = scaler.transform(input_df)\n",
        "\n",
        "    # Predict\n",
        "    prediction = model.predict(input_scaled)[0]\n",
        "    probability = model.predict_proba(input_scaled)[0][1]\n",
        "\n",
        "    result = \"Churn ❌\" if prediction == 1 else \"No Churn ✅\"\n",
        "\n",
        "    st.subheader(\"Prediction Result\")\n",
        "    st.write(f\"**Prediction:** {result}\")\n",
        "    st.write(f\"**Churn Probability:** {probability*100:.2f}%\")"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "abbdb465"
      },
      "source": [
        "pip install streamlit pandas numpy"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "756daba1"
      },
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pickle"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5667170b"
      },
      "source": [
        "# Task\n",
        "Create a Streamlit application and serve it from the `app.py` file."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9b24da3d"
      },
      "source": [
        "## create_streamlit_app_file\n",
        "\n",
        "### Subtask:\n",
        "Create a Python file named `app.py` containing the Streamlit application code.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4524544b"
      },
      "source": [
        "**Reasoning**:\n",
        "To create a Python file named `app.py` with the Streamlit application code, I will use the `%%writefile` magic command and populate it with the code from the previously executed cell.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9962ff4e"
      },
      "source": [
        "%%writefile app.py\n",
        "# app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import pickle\n",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "# ----------------------------\n",
        "# Title\n",
        "# ----------------------------\n",
        "st.title(\"📉 Customer Churn Prediction App\")\n",
        "st.write(\"Predict whether a customer will churn based on their details.\")\n",
        "\n",
        "# ----------------------------\n",
        "# Load Dataset\n",
        "# ----------------------------\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    df = pd.read_csv(\"/content/WA_Fn-UseC_-Telco-Customer-Churn.csv\")\n",
        "    return df\n",
        "\n",
        "data = load_data()\n",
        "st.write(\"Sample Dataset:\")\n",
        "st.dataframe(data.head())\n",
        "\n",
        "# ----------------------------\n",
        "# Preprocessing\n",
        "# ----------------------------\n",
        "@st.cache_data\n",
        "def preprocess_data(df):\n",
        "    df = df.copy()\n",
        "    # Example preprocessing (update according to your dataset)\n",
        "    df.fillna(0, inplace=True)\n",
        "\n",
        "    # Encode categorical columns\n",
        "    label_cols = df.select_dtypes(include=['object']).columns\n",
        "    le_dict = {}\n",
        "    for col in label_cols:\n",
        "        le = LabelEncoder()\n",
        "        df[col] = le.fit_transform(df[col])\n",
        "        le_dict[col] = le\n",
        "    return df, le_dict\n",
        "\n",
        "processed_data, le_dict = preprocess_data(data)\n",
        "\n",
        "# ----------------------------\n",
        "# Train Model (Random Forest)\n",
        "# ----------------------------\n",
        "@st.cache_resource\n",
        "def train_model(df):\n",
        "    X = df.drop(\"Churn\", axis=1)\n",
        "    y = df[\"Churn\"]\n",
        "\n",
        "    scaler = StandardScaler()\n",
        "    X_scaled = scaler.fit_transform(X)\n",
        "\n",
        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "    model.fit(X_scaled, y)\n",
        "\n",
        "    return model, scaler, X.columns\n",
        "\n",
        "model, scaler, feature_cols = train_model(processed_data)\n",
        "\n",
        "# ----------------------------\n",
        "# User Input for Prediction\n",
        "# ----------------------------\n",
        "st.sidebar.header(\"Enter Customer Details\")\n",
        "\n",
        "user_input = {}\n",
        "for col in feature_cols:\n",
        "    if col in le_dict:  # Categorical column\n",
        "        options = list(le_dict[col].classes_)\n",
        "        user_input[col] = st.sidebar.selectbox(col, options)\n",
        "    else:  # Numerical column\n",
        "        min_val = int(data[col].min())\n",
        "        max_val = int(data[col].max())\n",
        "        mean_val = int(data[col].mean())\n",
        "        user_input[col] = st.sidebar.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)\n",
        "\n",
        "# ----------------------------\n",
        "# Prediction\n",
        "# ----------------------------\n",
        "if st.sidebar.button(\"Predict Churn\"):\n",
        "    # Convert input to DataFrame\n",
        "    input_df = pd.DataFrame([user_input])\n",
        "\n",
        "    # Encode categorical inputs\n",
        "    for col in le_dict:\n",
        "        input_df[col] = le_dict[col].transform(input_df[col])\n",
        "\n",
        "    # Scale input\n",
        "    input_scaled = scaler.transform(input_df)\n",
        "\n",
        "    # Predict\n",
        "    prediction = model.predict(input_scaled)[0]\n",
        "    probability = model.predict_proba(input_scaled)[0][1]\n",
        "\n",
        "    result = \"Churn ❌\" if prediction == 1 else \"No Churn ✅\"\n",
        "\n",
        "    st.subheader(\"Prediction Result\")\n",
        "    st.write(f\"**Prediction:** {result}\")\n",
        "    st.write(f\"**Churn Probability:** {probability*100:.2f}%\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "998e9459"
      },
      "source": [
        "## run_streamlit_app\n",
        "\n",
        "### Subtask:\n",
        "Run the Streamlit application using `streamlit run app.py`.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5ca4fa9c"
      },
      "source": [
        "## Summary:\n",
        "\n",
        "### Data Analysis Key Findings\n",
        "\n",
        "*   A Python file named `app.py` was successfully created to house the Streamlit application code.\n",
        "*   The provided Streamlit application code, designed for customer churn prediction, was written into the `app.py` file.\n",
        "*   The file creation process completed without any errors.\n",
        "\n",
        "### Insights or Next Steps\n",
        "\n",
        "*   The next logical step is to run the newly created `app.py` file using Streamlit to verify its functionality and user interface.\n",
        "*   Consider testing the churn prediction model within the Streamlit app with various user inputs to ensure it performs as expected.\n"
      ]
    }
  ]
}

# Accessing Project Data with DVC

This project uses **[DVC (Data Version Control)](https://dvc.org/)** with **Google Drive remote storage** and a **service account** for seamless, authentication-free data access.

To download the project datasets (`dvc pull`), follow the secure setup below.

---

## 🛠 Prerequisites

Before you set up DVC, make sure you have the following tools installed step-by-step:

### 1. Clone the repository

```bash
git clone https://github.com/Pinminh/ml-arcene.git
cd ml-arcene
```

### 2. Create and activate virtual environment

```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

>Use `dvc --version` to comfirm that you have DVC installed.

---

## Step 1: Obtain the Service Account Credentials

Ask the project maintainer for the file: `gdrive-sa.json` 


This file contains the **Google service account credentials** required to access the shared Google Drive folder that stores DVC data.

> ⚠️ **Do not share this file publicly.**  
> Never commit it to Git or upload it to public cloud storage.

---

## Step 2: Place the File

Place the `gdrive-sa.json` file in the `root` of the project directory (same level as `.dvc` and `data`).


---

## Step 3: Configure DVC Locally

Run the following commands to configure DVC to use the service account **(locally only)**:

```bash
dvc remote modify --local gdrive_remote gdrive_use_service_account true
dvc remote modify --local gdrive_remote gdrive_service_account_json_file_path gdrive-sa.json
```
>These changes are saved in .dvc/config.local, which is Git-ignored for security.

---

## ⚙️ Step 4: Pull the Data

Once configured, pull the data using:

```bash
dvc pull
```
>This will download the dataset(s) tracked by DVC into your local project folder.
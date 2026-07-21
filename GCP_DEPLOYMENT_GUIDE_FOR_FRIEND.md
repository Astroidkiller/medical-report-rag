# 🚀 Google Cloud Run Deployment Guide
## Community Health Intelligence Assistant — Gen AI APAC Edition

This document provides exact, beginner-friendly instructions for deploying the project to Google Cloud Run using a Google Cloud account.

---

## 📋 Prerequisites
- A Google Cloud Account (with free trial credits or active billing)
- A web browser logged into Google Cloud Console ([console.cloud.google.com](https://console.cloud.google.com/))

---

## ⚡ Method 1: Deploying via Google Cloud Shell (Fastest — 3 Minutes)

No software installation is required on your local machine.

### Step 1: Open Cloud Shell
1. Log in to [Google Cloud Console](https://console.cloud.google.com/).
2. In the top-right toolbar, click the **Activate Cloud Shell** icon (`>_`).
3. A terminal pane will open at the bottom of your screen.

### Step 2: Clone Repository & Select Project
Run the following commands in Cloud Shell:

```bash
# 1. Set your GCP Project ID
gcloud config set project YOUR_PROJECT_ID

# 2. Clone the repository
git clone https://github.com/Astroidkiller/medical-report-rag.git
cd medical-report-rag
```

*(Replace `YOUR_PROJECT_ID` with your actual GCP Project ID shown at the top left of Google Cloud Console)*

### Step 3: Enable Google Cloud APIs
Run this command to activate all required Cloud Run, Build, and Vertex AI services:

```bash
gcloud services enable run.googleapis.com \
                       cloudbuild.googleapis.com \
                       containerregistry.googleapis.com \
                       aiplatform.googleapis.com
```

### Step 4: One-Click Build & Deploy
Execute the Cloud Build command:

```bash
gcloud builds submit --config=cloudbuild.yaml
```

**What happens next?**
- Cloud Build compiles the React frontend (`dist/`) and Python FastAPI backend into a production container.
- The container is deployed to **Google Cloud Run**.
- After 2–3 minutes, Cloud Shell will output your live URL:
  `https://community-health-assistant-xyz.a.run.app`

---

## 🔑 Step 5: Adding Environment Variables (Optional for Full Google AI Mode)

To enable **Google Maps Platform** and **Vertex AI / Gemini 1.5** in production:

1. Open [Google Cloud Run Console](https://console.cloud.google.com/run).
2. Click on the service name: **`community-health-assistant`**.
3. Click **Edit & Deploy New Revision**.
4. Scroll down to **Variables & Secrets** → Click **Add Variable**:
   - `GEMINI_API_KEY`: *(your API key from Google AI Studio)*
   - `VITE_GOOGLE_MAPS_API_KEY`: *(your key from Google Maps Platform)*
   - `LLM_PROVIDER`: `vertex_ai`
   - `GCP_PROJECT_ID`: `YOUR_PROJECT_ID`
   - `GCP_LOCATION`: `us-central1`
5. Click **Deploy**.

---

## 🔒 Security & Architecture Safeguards

- **Zero-Crash Architecture**: The app uses automatic fallback adapters. If any GCP key is omitted, it gracefully defaults to local/standard Gemini API mode.
- **Cost**: Cloud Run has a free tier of **2 million requests/month**. Running this demo will cost **$0.00**.
- **Security**: Raw ChromaDB has been isolated using an in-memory vector engine, resolving all security flags.

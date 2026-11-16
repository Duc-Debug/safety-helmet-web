# Safety Vision AI - React Frontend

Modern React + Vite + TypeScript + TailwindCSS SPA for Safety Helmet & PPE Compliance Monitoring with Rule-Based XAI reasoning.

---

## 🚀 Features

- **Dashboard Page**:
  - Image upload with drag & drop preview.
  - Model architecture selector (**YOLOv8s** fine-tuned, **YOLOv8n** lightweight, **YOLOv5** baseline).
  - Compact active parameters indicator (`Person: 0.35`, `PPE: 0.25`, `IoU: 0.45`).
  - Interactive bounding box viewer & Rule Evaluation Trace Table (R01–R05 forward chaining).
- **Comparison Page**:
  - Runs all 3 models concurrently on the same input image.
  - Side-by-side metric comparison: Inference speed (ms), Worker count, PPE detection, violations, and recommendation summary.
- **Tuning Studio**:
  - Interactive sliders to tune confidence thresholds (`conf_person`, `conf_ppe`, `iou_threshold`).
  - Direct live testing of edge cases (occlusion, crowded environments).

---

## 🛠️ Local Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure API base URL:**
   By default, `.env` points to `http://localhost:8000`:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

4. **Build production bundle:**
   ```bash
   npm run build
   ```

---

## 🌐 Deploy to Vercel

1. Push your repository to GitHub / GitLab.
2. Log in to [Vercel](https://vercel.com) and click **"Add New Project"**.
3. Select your repository:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Set Environment Variable in Vercel Project Settings:
   - `VITE_API_BASE_URL`: URL of your deployed FastAPI backend (e.g., Render, Railway, Hugging Face Spaces, or ngrok tunnel).
5. Click **Deploy**.

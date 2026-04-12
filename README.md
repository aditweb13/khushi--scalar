---
title: SmartRecco-Env
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
---
# SmartRecco-Env: OpenEnv E-commerce Recommendation Agent

SmartRecco-Env is an **OpenEnv-compliant reinforcement learning environment** that simulates real-world **e-commerce product recommendation curation**. An AI agent acts as a recommendation curator, tasked with analyzing user profiles, filtering a product catalog, and curating personalized recommendation lists.

This environment addresses a highly practical, non-toy industry problem.

> **Note:** The original full-stack Django/Next.js implementation is preserved in `smartrecco-backend/` and `smartreco-frontend/`.

## 🚀 Environment Overview

The environment evaluates agents based on multi-step reasoning, constraint satisfaction, and personalization.

### Action Space (`models.Action`)
Agents interact using typed Pydantic models with the following action types:
*   `browse`: Browse products by category.
*   `select`: Select product IDs for the recommendation list.
*   `submit`: Submit the final list for grading.
*   `ask_preference`: (Cold-start only) Ask the user a question to reveal hidden preferences.

### Observation Space (`models.Observation`)
After each action, the agent receives:
*   `user_profile`: Demographics, budget, and known interests.
*   `products_visible`: Products currently browsed.
*   `current_selections`: Products currently selected.
*   `budget_remaining`, `message`, and `feedback` (if any).

## 🏆 Tasks & Difficulty Tiers

The environment defines 3 progressive tasks in `openenv.yaml`:

1.  **Easy (`easy_single_category`)**: Recommend 5 products to a user with known interests. Focuses purely on relevance.
2.  **Medium (`medium_multi_constrained`)**: Recommend 8 products across categories. Focuses on balancing relevance, a strict budget cap ($200), and diversity.
3.  **Hard (`hard_coldstart_adaptive`)**: Cold-start user. Agent must query the user to uncover preferences, then satisfy complex constraints before submitting.

### Reward Function & Grading
The `graders/grader.py` provides a partial-credit score [0.0 - 1.0] based on four factors:
1.  **Relevance** (40%): Overlap between selected product tags and user interests.
2.  **Diversity** (20%): Category spread among selections.
3.  **Constraints** (20%): Adherence to budget and item count targets.
4.  **Ranking** (20%): Price-based coherence.

## ⚙️ Setup & Installation

### Local Setup
```bash
# Set up a python environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate # or source venv/bin/activate
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running the Inference Script
The inference script operates via the OpenAI Python Client as per the Hackathon mandatory instructions. 

Provide your environment credentials and execute:

```bash
# Export mandatory environment parameters
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" 
export HF_TOKEN="hf_your_token_here"

# Run the inference script (logs output strictly in [START]/[STEP]/[END] format)
python inference.py
```

## 🐳 Deployment (Hugging Face Spaces)
This environment is ready for deployment to HF Spaces.
1. Create a Docker-based Space on Hugging Face.
2. Upload this repository.
3. The included `Dockerfile` will automatically serve the environment on port 7860.

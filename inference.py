import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import EnvClient
from models import Action
from graders.grader import RecommendationGrader

# Mandatory Environment Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "smartrecco_env"

MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.5  # Normalized score [0, 1] required for task "success"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Replace newlines in action string to ensure single line output
    safe_action = action.replace('\n', ' ').replace('\r', '')
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_openenv_inference():
    tasks = ["easy_single_category", "medium_multi_constrained", "hard_coldstart_adaptive"]
    
    if not API_KEY:
        print("Warning: HF_TOKEN or API_KEY not set", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient()

    for task_id in tasks:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False
        
        try:
            init_data = env.reset(task_id)
            obs = init_data["observation"]
            
            system_prompt = textwrap.dedent("""
            You are an AI Recommendation Agent for an e-commerce platform.
            Your goal is to curate a list of personalized products for the user.
            You must reply with a valid JSON action conforming to this schema:
            {
              "action_type": "browse" | "select" | "submit" | "ask_preference",
              "category": "String (optional)",
              "product_ids": [integer] (optional),
              "query": "String (optional)"
            }
            Output raw JSON ONLY.
            """).strip()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(obs)}
            ]

            done = False
            
            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                # Get action from LLM
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.0
                    )
                    action_text = (response.choices[0].message.content or "").strip()
                    if action_text.startswith("```json"): action_text = action_text[7:-3]
                    elif action_text.startswith("```"): action_text = action_text[3:-3]
                    
                    action_dict = json.loads(action_text)
                    action = Action(**action_dict)
                    error = None
                except Exception as e:
                    action = Action(action_type="submit")
                    action_text = '{"action_type": "submit"}'
                    error = str(e).replace('\n', ' ')

                # Execute step
                try:
                    step_result = env.step(action)
                    obs = step_result["observation"]
                    reward = step_result["reward"] or 0.0
                    done = step_result["done"]
                    
                    messages.append({"role": "assistant", "content": action_text})
                    messages.append({"role": "user", "content": json.dumps(obs)})
                except Exception as e:
                    reward = 0.0
                    done = True
                    error = f"Env step logic failed: {str(e)}"
                    obs = {}
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_text, reward=reward, done=done, error=error)

            # Get final score
            final_state = env.state()
            score = RecommendationGrader.grade(final_state)
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            print(f"[DEBUG] Error running inference: {e}", flush=True)
            
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    run_openenv_inference()

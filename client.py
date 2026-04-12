import requests
from models import Action
from typing import Dict, Any

class EnvClient:
    def __init__(self, server_url: str = "http://localhost:7860"):
        self.server_url = server_url

    def reset(self, task_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.server_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Action) -> Dict[str, Any]:
        resp = requests.post(f"{self.server_url}/step", json=action.dict())
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.server_url}/state")
        resp.raise_for_status()
        return resp.json()

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class ProductInfo(BaseModel):
    id: int
    name: str
    category: str
    price: float
    tags: List[str]

class UserProfile(BaseModel):
    user_id: int
    name: str
    age: int
    interests: List[str]
    budget: float
    knowledge: str = "Hidden unless preferences are asked"

class Action(BaseModel):
    action_type: Literal["browse", "select", "submit", "ask_preference"]
    category: Optional[str] = Field(None, description="Category to browse")
    product_ids: Optional[List[int]] = Field(None, description="Products to select for recommendation")
    query: Optional[str] = Field(None, description="Question to ask user for cold-start")

class Observation(BaseModel):
    step: int
    products_visible: List[ProductInfo]
    user_profile: UserProfile
    current_selections: List[ProductInfo]
    feedback: Optional[str] = None
    budget_remaining: float
    constraints_met: bool = False
    message: str = ""

class State(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    user_profile: UserProfile
    catalog: List[ProductInfo]
    selected_products: List[int]
    submitted: bool
    reward: float
    done: bool
    score_breakdown: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

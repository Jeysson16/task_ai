from fastapi import FastAPI
from pydantic import BaseModel
from src.llm_service import ask_llm

app = FastAPI()

class Context(BaseModel):
    user_id: str
    goal_id: str
    goal_priority: int
    goal_progress: float
    scheduled_hour: int
    available_morning: int
    available_afternoon: int
    available_evening: int
    has_emergency: int
    recent_tasks: list[str]

@app.post("/recommend/")
def recommend(ctx: Context):
    prompt = build_prompt(ctx)
    completion = ask_llm(prompt, max_tokens=128)
    # parse JSON-like output or texto estructurado
    return {"recommendation": completion}

def build_prompt(ctx: Context) -> str:
    # Aquí construyes un prompt con few-shot examples y el estado actual
    examples = """
    Example:
    User: priority=5, progress=0.2, hour=14, avail_m=1, a_m=0, e=0, tasks=[...]
    Assistant: \"\"\"{{"action":"reschedule_later_today","hour":16}}\"\"\"
    """
    state = f"""
    user_id: {ctx.user_id}
    goal_id: {ctx.goal_id}
    priority: {ctx.goal_priority}
    progress: {ctx.goal_progress:.2f}
    scheduled_hour: {ctx.scheduled_hour}
    avail: [m={ctx.available_morning}, a={ctx.available_afternoon}, e={ctx.available_evening}]
    emergency: {ctx.has_emergency}
    recent_tasks: {ctx.recent_tasks}
    """
    return examples + "\nNow decide the best recommendation (as JSON):\n" + state
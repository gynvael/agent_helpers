import json
from .models import *

# https://openai.com/pricing
PRICING = {
  MODEL_GPT35:     (0.0015, 0.0020),  # input, output, per 1K tokens
  MODEL_GPT4:      (0.0300, 0.0600),
  MODEL_GPT4TURBO: (0.0100, 0.0300),
}


def update_billing(model, prompt_tokens, completion_tokens):
  try:
    with open("billing.json", "r") as f:
      billing = json.load(f)
  except FileNotFoundError:
    billing = {
      "total_cost": 0.00,  # Yeah, double, don't care.
    }

  pricing = PRICING[model]

  cost_increase = (prompt_tokens * pricing[0]) / 1000
  cost_increase += (completion_tokens * pricing[1]) / 1000

  billing["total_cost"] += cost_increase

  with open("billing.json", "w") as f:
    json.dump(billing, f)

  with open("billing_history.csv", "a") as f:
    f.write(
        f"{model},{prompt_tokens},{completion_tokens},{billing['total_cost']}\n"
    )

  return cost_increase, billing["total_cost"]

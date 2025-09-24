import json
import pandas as pd
from openai import OpenAI
import numpy as np




client = OpenAI(api_key='*****************************')

# Prompt for synthetic Huberman-inspired fitness data
prompt = """
Generate synthetic 30-day fitness and recovery log data inspired by Andrew Huberman protocols.
Include:
- Date
- Workout type (resistance, cardio, mobility, rest)
- Muscle group (if resistance training)
- Sets
- Reps
- Weight (kg) [if applicable]
- Sleep hours
- Morning sunlight exposure (minutes)
- HRV score
- Focus level (1-10)
- Notes (e.g. caffeine intake, cold plunge, sauna)

Return the output in JSON format.
"""

# Generate response in JSON format
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)

# Load into DataFrame
df = pd.DataFrame(json.loads(response.choices[0].message.content))
print(df.head())

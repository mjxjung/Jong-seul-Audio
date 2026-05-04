"""Query bank — same questions are spoken by every character.

Kept short and conversational so TTS sounds natural and an Audio-LM can answer
in 1–4 sentences.
"""

QUERIES = [
    {"id": "q_001", "category": "advice",
     "text": "I've been feeling really stressed about work lately. What should I do?"},
    {"id": "q_002", "category": "recommendation",
     "text": "What's a good book I should read this month?"},
    {"id": "q_003", "category": "information",
     "text": "Can you explain how machine learning works?"},
    {"id": "q_004", "category": "opinion",
     "text": "What do you think about working from home versus going to the office?"},
    {"id": "q_005", "category": "recommendation",
     "text": "I want to start exercising. Where should I begin?"},
    {"id": "q_006", "category": "advice",
     "text": "I had an argument with my best friend. How do I make things right?"},
    {"id": "q_007", "category": "information",
     "text": "Why do leaves change color in autumn?"},
    {"id": "q_008", "category": "recommendation",
     "text": "What should I cook for dinner tonight?"},
]

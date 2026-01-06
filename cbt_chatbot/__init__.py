"""
CBT Therapy Chatbot MVP

A verifiable CBT chatbot that:
1. Identifies how a user feels (emotion + intensity)
2. Identifies why they feel that way (situation → thought → feeling chain)
3. Detects dysfunctional thinking patterns (cognitive distortions)
4. Detects unproductive behaviors
5. Never misses a crisis (hard safety constraint)
6. Intervenes appropriately when we have enough information

Architecture: LLM as sensor, not decision-maker
- Extraction: LLM → Structured data
- State Update: Pure Python
- Policy Decision: Pure Python (formal rules)
- Response Generation: LLM (within constraints)
- Output Validation: Hard safety gate
"""

__version__ = "0.1.0"

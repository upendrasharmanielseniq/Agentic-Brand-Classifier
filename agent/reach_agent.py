import dspy

class ReachEstimatorAgent:
    """
    Agent that calculates estimated reach for a given campaign.
    """

    def __init__(self, llm=None):
        self.llm = llm or dspy.LM("ollama/phi3", api_base="http://localhost:11434")

    def calculate_reach(self, campaign_id: str):
        # Deterministic numeric metric derived from campaign_id
        # (using hash for reproducibility)
        seed = abs(hash(campaign_id)) % 10_000
        estimated_reach = 5000 + (seed % 95000)  # Range: 5k–100k
        
        message = f"Campaign {campaign_id} has an estimated reach of {estimated_reach} users."
        return {
            "campaign_id": campaign_id,
            "estimated_reach": estimated_reach,
            "message": message
        }

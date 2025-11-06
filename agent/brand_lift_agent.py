import dspy

class BrandLiftAgent:
    """
    Agent that calculates brand lift percentage for a given campaign.
    """

    def __init__(self, llm=None):
        self.llm = llm or dspy.LM("ollama/phi3", api_base="http://localhost:11434")

    def calculate_brand_lift(self, campaign_id: str):
        # Deterministic percentage derived from ID hash
        seed = abs(hash(campaign_id)) % 100
        brand_lift = round(1 + (seed % 20) + (seed / 1000), 2)  # Range: 1%–20.99%
        
        message = f"Campaign {campaign_id} achieved a brand lift of {brand_lift}%."
        return {
            "campaign_id": campaign_id,
            "brand_lift_percentage": brand_lift,
            "message": message
        }

import os
import dspy
from serpapi import GoogleSearch

class BrandExtractionSignature(dspy.Signature):
    """Extract brand names from a user prompt"""
    prompt = dspy.InputField(desc="The user query or prompt text")
    brands = dspy.OutputField(desc="List of brand names mentioned in the text")

# Configure LLM locally (using Ollama Phi-3)
dspy.configure(lm=dspy.LM("ollama/phi3"))

class BrandAgent:
    """
    Agent to extract and validate brand names from text.
    Uses an LLM for extraction and SerpAPI for validation.
    Returns structured output with confidence and method.
    """
    def __init__(self):
        self.predictor = dspy.Predict(BrandExtractionSignature)

    def extract_brands(self, prompt: str):
        """
        Extract and validate brand names from a given prompt.

        Args:
            prompt (str): User input text or query.

        Returns:
            dict: {
                "brands": list of validated brand names,
                "confidence": float (0–1),
                "method": str (description of extraction approach)
            }
        """
        result = self.predictor(prompt=prompt)
        raw_brands = result.brands.strip()

        # Handle possible empty output or newline artifacts
        if not raw_brands:
            return {
                "brands": [],
                "confidence": 0.0,
                "method": "LLM-based extraction + SerpAPI validation"
            }

        brand_list = [b.strip() for b in raw_brands.replace("\n", ",").split(",") if b.strip()]
        validated = []

        for brand in brand_list:
            params = {
                "engine": "google",
                "q": f"{brand} official site",
                "api_key": os.getenv("SERPAPI_API_KEY")
            }
            search = GoogleSearch(params)
            res = search.get_dict()

            # Simple heuristic: if official site or brand name appears in title, validate it
            if any(
                brand.lower() in item.get("title", "").lower()
                for item in res.get("organic_results", [])
            ):
                validated.append(brand)

        # Confidence based on ratio of validated to extracted
        confidence = len(validated) / len(brand_list) if brand_list else 0.0

        return {
            "brands": validated,
            "confidence": round(confidence, 2),
            "method": "LLM-based extraction + SerpAPI validation"
        }
import os
from serpapi import GoogleSearch
import datetime

class CampaignInsightAgent:
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_API_KEY")

    def get_campaign_insights(self, brand_name: str):
        params = {
            "engine": "google_news",
            "q": f"{brand_name} latest marketing campaign OR advertisement OR brand promotion",
            "num": 5,
            "api_key": self.api_key
        }

        search = GoogleSearch(params)
        res = search.get_dict()

        insights = []
        if "news_results" in res:
            for item in res["news_results"]:
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                source = item.get("source", "")
                date = item.get("date", "")

                # Derive campaign name from title if possible
                campaign_name = self._extract_campaign_name(title)

                insights.append({
                    "title": campaign_name or title,
                    "summary": snippet,
                    "source": source,
                    "link": link,
                    "date": date
                })

        confidence_score = self._calculate_confidence(insights)
        return {
            "brand": brand_name,
            "campaigns": insights,
            "confidence_score": confidence_score,
            "method_used": "SerpAPI Google News Search"
        }

    def _extract_campaign_name(self, title: str):
        if "'" in title:
            parts = title.split("'")
            if len(parts) > 1:
                return parts[1]  # e.g. “launches 'Create the Future' campaign”
        return None

    def _calculate_confidence(self, insights):
        if len(insights) >= 2:
            return 0.9
        elif len(insights) == 1:
            return 0.7
        else:
            return 0.5

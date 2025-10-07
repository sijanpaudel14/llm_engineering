from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent

from dotenv import load_dotenv
load_dotenv(override=True)
import os
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY_1")
class ScannerAgent(Agent):
    """
    ScannerAgent implemented using LangChain. Fetches deals from RSS feeds
    and selects the top 5 with the most detailed description and clear price.
    """

    MODEL = "gemini-2.5-flash"  # Or "gemini-2.0-flash" for Google
    SYSTEM_PROMPT = """You are a highly analytical AI that extracts the 5 best deals from a list. Your most critical task is to distinguish between a final price and a discount amount.

You will select deals based on two criteria:
1.  A detailed, high-quality product description.
2.  A clear, unambiguous, and final price.

CRUCIAL RULE ON PRICING: The price you extract MUST be the final, total cost of the item. It is never a discount amount.
-   If a deal mentions a price using words like "$XXX off", "save $XXX", "reduced by $XXX", or "a $XXX discount", you MUST IGNORE AND DISCARD that deal completely.
-   You must look for explicit pricing clues like "Price:", "Costs:", "For:", or a standalone number that clearly represents the total sale price.
-   If you have ANY doubt about whether a number is a final price or a discount, DO NOT include the deal. It is better to return fewer than 5 deals than to include one with an incorrect price.

For the 5 deals you select, respond strictly in JSON with no explanation. The `product_description` should be a summary of the item itself, not the deal's terms.

{{"deals": [
    {{
        "product_description": "Your clearly expressed summary of the product in 4-5 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a paragraph of text for each item you choose.",
        "price": 99.99,
        "url": "the url as provided"
    }},
    ...
]}}"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.

Deals:
"""

    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ScannerAgentLC with a LangChain chat model.

        :param api_key: Optional API key for OpenAI or Google LLMs
        """
        self.log("Scanner Agent is initializing (LangChain)")
        self.llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash"
        )
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory: List[ScrapedDeal]) -> List[ScrapedDeal]:
        """
        Retrieve all deals from RSS feeds that are not already in memory.

        :param memory: List of previously seen ScrapedDeal objects
        :return: List of new ScrapedDeal objects
        """
        self.log("Scanner Agent fetching deals")
        urls = [d.url for d in memory]
        scraped = ScrapedDeal.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent found {len(result)} new deals")
        return result

    def make_user_prompt(self, scraped: List[ScrapedDeal]) -> ChatPromptTemplate:
        """
        Create a user prompt for the LLM based on the scraped deals.

        :param scraped: List of ScrapedDeal objects
        :return: ChatPromptTemplate containing system and user messages
        """
        user_content = '\n\n'.join([scrape.describe() for scrape in scraped])
        full_prompt = self.USER_PROMPT_PREFIX + user_content + self.USER_PROMPT_SUFFIX
        return ChatPromptTemplate([
            ("system", self.SYSTEM_PROMPT),
            ("user", full_prompt)
        ])

    def scan(self, memory: List[ScrapedDeal] = []) -> Optional[DealSelection]:
        """
        Call the LLM to select the top 5 deals with detailed descriptions and clear prices.

        :param memory: List of ScrapedDeal objects already seen
        :return: DealSelection object containing the selected deals or None
        """
        scraped = self.fetch_deals(memory)
        if not scraped:
            return None

        prompt = self.make_user_prompt(scraped)
        self.log("Scanner Agent calling LLM (LangChain)")

        # Generate response using LangChain
        response = self.llm(prompt.format_prompt().to_messages())

        # Parse structured output
        try:
            result = DealSelection.parse_raw(response.content)
            result.deals = [deal for deal in result.deals if deal.price > 0]
            self.log(f"Scanner Agent received {len(result.deals)} deals with price>0")
            return result
        except Exception as e:
            self.log(f"Error parsing deals: {e}")
            return None

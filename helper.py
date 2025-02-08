import asyncio
import json
import os
import subprocess
import urllib
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
import re
from bs4 import BeautifulSoup
from gtts import gTTS


# ============================ CHATBOT CLASS ============================

class ChatBot:
    """
    A chatbot class that interacts with a local Llama model using Ollama.
    """

    def __init__(self) -> None:
        """Initialize the ChatBot instance with a conversation history."""
        self.history: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the chatbot based on the user's prompt.

        Args:
            prompt (str): The input message from the user.

        Returns:
            str: The chatbot's response to the provided prompt.
        """
        self.history.append({"role": "user", "content": prompt})

        # Convert chat history into a string for subprocess input
        conversation = "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)

        try:
            # Run the Llama model using Ollama
            completion = subprocess.run(
                ["ollama", "run", "llama3.2:latest"],
                input=conversation,
                capture_output=True,
                text=True,
            )

            if completion.returncode != 0:
                print("Error:", completion.stderr)
                return "I'm sorry, I encountered an issue processing your request."

            response = completion.stdout.strip()
            self.history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            print(f"Error sending query to the model: {e}")
            return "I'm sorry, an error occurred while processing your request."

    async def rate_body_of_article(self, article_title: str, article_content: str) -> str:
        """
        Rate the quality of an article's content based on its title.

        Args:
            article_title (str): The title of the article.
            article_content (str): The full content of the article.

        Returns:
            str: A rating between 1 and 5 based on relevance and quality.
        """
        prompt = f"""
        Given the following article title and content, provide a rating between 1 and 5 
        based on how well the content aligns with the title and its overall quality.

        - **Article Title**: {article_title}
        - **Article Content**: {article_content[:1000]}  # Limit to first 1000 chars

        **Instructions:**
        - The rating should be a whole number between 1 and 5.
        - Base your score on accuracy, clarity, and relevance.
        - Only return a single numeric value (1-5) with no extra text.

        **Example Output:**
        `4`
        """

        try:
            # Run the Llama model using Ollama
            completion = subprocess.run(
                ["ollama", "run", "llama3.2:latest"],
                input=prompt,
                capture_output=True,
                text=True,
            )

            if completion.returncode != 0:
                print("Error:", completion.stderr)
                return "Error"

            response = completion.stdout.strip()

            # Validate the rating is within the expected range
            if response.isdigit() and 1 <= int(response) <= 5:
                self.history.append({"role": "assistant", "content": response})
                return response
            else:
                return "Error"

        except Exception as e:
            print(f"Error sending query to the model: {e}")
            return "Error"


# ============================ EXTRACT NEWS BODY ============================

def extract_news_body(news_url: str) -> str:
    """
    Extract the full article body from a given news URL.

    Args:
        news_url (str): The URL of the news article.

    Returns:
        str: Extracted full article content.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

        response = requests.get(news_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return "Failed to fetch article."

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")

        # Extract and return cleaned text
        return "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

    except Exception as e:
        return f"Error extracting article content: {e}"


# ============================ ASYNC NEWS SCRAPING ============================

async def invoke_duckduckgo_news_search(query: str, num: int = 5, location: str = "us-en", time_filter: str = "w") -> \
Dict[str, Any]:
    """
    Perform a DuckDuckGo News search, extract news headlines, fetch full content,
    and rate articles using parallel asynchronous processing.

    Args:
        query (str): The search query string.
        num (int): Number of search results to retrieve.
        location (str): The region code for location-based results (e.g., 'us-en', 'in-en').
        time_filter (str): Time filter for news ('d' = past day, 'w' = past week, 'm' = past month, 'y' = past year).

    Returns:
        Dict[str, Any]: A dictionary containing extracted news articles.
    """

    duckduckgo_news_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}&kl={location}&df={time_filter}&ia=news"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(duckduckgo_news_url, headers=headers)
    if response.status_code != 200:
        return {"status": "error", "message": "Failed to fetch news search results"}

    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("div", class_="result__body")

    async def process_article(result, index: int) -> Optional[Dict[str, Any]]:
        """Processes a single article: extracts details, fetches content, and rates it."""
        try:
            title_tag = result.find("a", class_="result__a")
            if not title_tag:
                return None

            title = title_tag.text.strip()
            raw_link = title_tag["href"]

            match = re.search(r"uddg=(https?%3A%2F%2F[^&]+)", raw_link)
            link = urllib.parse.unquote(match.group(1)) if match else "Unknown Link"

            snippet_tag = result.find("a", class_="result__snippet")
            summary = snippet_tag.text.strip() if snippet_tag else "No summary available."

            article_content = extract_news_body(link)

            bot = ChatBot()
            rating = await bot.rate_body_of_article(title, article_content)

            return {
                "num": index + 1,
                "link": link,
                "title": title,
                "summary": summary,
                "body": article_content,
                "rating": rating
            }

        except Exception as e:
            print(f"[DEBUG] Error processing article: {e}")
            return None

    tasks = [process_article(result, index) for index, result in enumerate(search_results[:num])]
    extracted_results = await asyncio.gather(*tasks)

    extracted_results = [res for res in extracted_results if res is not None]

    return {"status": "success", "results": extracted_results} if extracted_results else {
        "status": "error", "message": "No valid news search results found"}


# ============================ UTILITY FUNCTIONS ============================

def current_year() -> int:
    """Returns the current year as an integer."""
    return datetime.now().year


def save_to_audio(text: str) -> None:
    """Converts text to an audio file using Google Text-to-Speech (gTTS)."""
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")


# ============================ TEST EXECUTION ============================

if __name__ == "__main__":
    query = "latest AI breakthroughs"
    results = asyncio.run(invoke_duckduckgo_news_search(query, num=10, location="us-en", time_filter="w"))
    print(json.dumps(results, indent=2))

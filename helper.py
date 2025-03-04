import asyncio
import json
import os
import pickle
import subprocess
import time
import urllib.parse
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
import keras
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from gtts import gTTS
from huggingface_hub import hf_hub_download
from keras.utils import pad_sequences
from transformers import BertTokenizer

from logger.app_logger import app_logger
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import concurrent.futures

class ChatBot:
    """
    A chatbot class that interacts with a local Llama model using Ollama.
    """

    def __init__(self) -> None:
        """Initialize the ChatBot instance with a conversation history."""
        self.history: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]
        app_logger.log_info("ChatBot instance initialized", level="INFO")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the chatbot based on the user's prompt.

        Args:
            prompt (str): The input message from the user.

        Returns:
            str: The chatbot's response to the provided prompt.
        """
        self.history.append({"role": "user", "content": prompt})
        app_logger.log_info("User prompt added to history", level="INFO")

        # Convert chat history into a string for subprocess input
        conversation: str = "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)

        try:
            # Run the Llama model using Ollama
            completion: subprocess.CompletedProcess = subprocess.run(
                ["ollama", "run", "llama3.2:latest"],
                input=conversation,
                capture_output=True,
                text=True,
            )

            if completion.returncode != 0:
                app_logger.log_error(f"Error running subprocess: {completion.stderr}")
                return "I'm sorry, I encountered an issue processing your request."

            response: str = completion.stdout.strip()
            self.history.append({"role": "assistant", "content": response})
            app_logger.log_info("Assistant response generated", level="INFO")

            return response

        except Exception as e:
            app_logger.log_error(f"Error sending query to the model: {e}")
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
        prompt: str = f"""
        Given the following article title and content, provide a rating between 1 and 5
        based on how well the content aligns with the title and its overall quality.

        - **Article Title**: {article_title}
        - **Article Content**: {article_content[:1000]}  # Limit to first 1000 chars

        **Instructions:**
        - The rating should be a whole number between 1 and 5.
        - Base your score on accuracy, clarity, and relevance.
        - Only return a single numeric value (1-5) with no extra text.

        **Example Output:**
        `4` or `2` or `3.5` or `1.5`
        """

        try:
            # Run the Llama model using Ollama
            completion: subprocess.CompletedProcess = subprocess.run(
                ["ollama", "run", "llama3.2:latest"],
                input=prompt,
                capture_output=True,
                text=True,
            )

            if completion.returncode != 0:
                app_logger.log_error(f"Error running subprocess: {completion.stderr}")
                return "Error"

            response: str = completion.stdout.strip()

            # Validate the rating is within the expected range
            if response.isdigit() and 1 <= int(response) <= 5:
                self.history.append({"role": "assistant", "content": response})
                app_logger.log_info(f"Article rated: {response}", level="INFO")
                return response
            else:
                app_logger.log_warning(f"Invalid rating received: {response}")
                return "Error"

        except Exception as e:
            app_logger.log_error(f"Error sending query to the model: {e}")
            return "Error"

    async def rate_article_credibility(self, article_title: str, article_content: str) -> str:
        """
        Rate the credibility of an article using a locally created model.

        Args:
            article_title (str): The title of the article.
            article_content (str): The full content of the article.

        Returns:
            str: A credibility rating based on the model's prediction.
        """
        try:
            # Load the model
            model_path: str = hf_hub_download(repo_id="Dkethan/my-tf-nn-model-v2", filename="model.keras")
            new_model = keras.models.load_model(model_path)

            # Load the Hugging Face tokenizer
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            # Preprocess the input data
            max_length: int = new_model.input_shape[0][1]  # Ensure max_length matches the model input
            X_text = tokenizer(
                [article_title],  # Tokenize the article title
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="tf"
            )

            # Dummy 'func_rating' input (can be replaced with actual data)
            X_func_rating: np.ndarray = np.array([5]).reshape(-1, 1)  # Replace with actual input if available

            # Make predictions
            predictions: np.ndarray = new_model.predict(
                {"text_input": X_text["input_ids"], "func_rating_input": X_func_rating}
            )
            prediction: int = np.argmax(predictions, axis=1)[0]

            # Log and return the prediction
            app_logger.log_info(f"Article credibility rated: {prediction}", level="INFO")
            return str(prediction)

        except Exception as e:
            app_logger.log_error(f"Error rating article credibility: {e}")
            return "Error"


def extract_news_body(news_url: str) -> str:
    """
    Extract the full article body from a given news URL.

    Args:
        news_url (str): The URL of the news article.

    Returns:
        str: Extracted full article content.
    """
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    retries: int = 3
    for attempt in range(retries):
        try:
            response: requests.Response = requests.get(news_url, headers=headers, timeout=10)
            if response.status_code == 403:
                app_logger.log_error(f"Access forbidden to article: {response.status_code}")
                return "Access forbidden to article."
            if response.status_code != 200:
                app_logger.log_error(f"Failed to fetch article: {response.status_code}")
                return "Failed to fetch article."

            soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
            paragraphs: List[BeautifulSoup] = soup.find_all("p")

            # Extract and return cleaned text
            article_content: str = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
            app_logger.log_info(f"Article content extracted from {news_url}", level="INFO")
            return article_content

        except requests.exceptions.Timeout:
            app_logger.log_warning(f"Timeout occurred while fetching article: {news_url}, attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retrying
                continue
            return "Error: Timeout occurred while fetching article."

        except Exception as e:
            app_logger.log_error(f"Error extracting article content: {e}")
            return f"Error extracting article content: {e}"

    return "Failed to fetch article after multiple attempts."

async def invoke_duckduckgo_news_search(query: str, num: int = 3, location: str = "us-en", time_filter: str = "w") -> Dict[str, Any]:
    """
    Perform a news search on DuckDuckGo and return the results.

    Args:
        query (str): The search query.
        num (int): The number of results to return.
        location (str): The location filter for the search.
        time_filter (str): The time filter for the search.

    Returns:
        Dict[str, Any]: A dictionary containing the search results.
    """
    app_logger.log_info(f"Starting DuckDuckGo news search for query: {query}", level="INFO")

    chrome_options: Options = Options()
    chrome_options.add_argument("--headless")
    driver: webdriver.Chrome = webdriver.Chrome(options=chrome_options)

    duckduckgo_news_url: str = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}&kl={location}&df={time_filter}&ia=news"
    driver.get(duckduckgo_news_url)

    soup: BeautifulSoup = BeautifulSoup(driver.page_source, "html.parser")
    search_results: List[BeautifulSoup] = soup.find_all("div", class_="result__body")

    def process_article(result: BeautifulSoup, index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single search result and extract relevant information.

        Args:
            result (BeautifulSoup): The search result to process.
            index (int): The index of the search result.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the extracted information, or None if an error occurs.
        """
        try:
            title_tag: Optional[BeautifulSoup] = result.find("a", class_="result__a")
            if not title_tag:
                app_logger.log_warning(f"Title tag not found for result index {index}")
                return None

            title: str = title_tag.text.strip()
            raw_link: str = title_tag["href"]

            match: Optional[re.Match] = re.search(r"uddg=(https?%3A%2F%2F[^&]+)", raw_link)
            link: str = urllib.parse.unquote(match.group(1)) if match else "Unknown Link"

            snippet_tag: Optional[BeautifulSoup] = result.find("a", class_="result__snippet")
            summary: str = snippet_tag.text.strip() if snippet_tag else "No summary available."

            article_content: str = extract_news_body(link)

            bot: ChatBot = ChatBot()

            # Rate the rate_body_of_article
            # rating: str = asyncio.run(bot.rate_body_of_article(title, article_content))

            # Rate the credibility of the article
            rating: str = asyncio.run(bot.rate_article_credibility(title, article_content))

            app_logger.log_info(f"Processed article: {title}", level="INFO")

            return {
                "num": index + 1,
                "link": link,
                "title": title,
                "summary": summary,
                "body": article_content,
                "rating": rating
            }

        except Exception as e:
            app_logger.log_error(f"Error processing article: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks: List[concurrent.futures.Future] = [executor.submit(process_article, result, index) for index, result in enumerate(search_results[:num])]
        extracted_results: List[Optional[Dict[str, Any]]] = [task.result() for task in concurrent.futures.as_completed(tasks)]

    driver.quit()

    extracted_results = [res for res in extracted_results if res is not None]

    if extracted_results:
        app_logger.log_info(f"News search completed successfully with {len(extracted_results)} results", level="INFO")
        return {"status": "success", "results": extracted_results}
    else:
        app_logger.log_error("No valid news search results found")
        return {"status": "error", "message": "No valid news search results found"}

def current_year() -> int:
    """Returns the current year as an integer."""
    return datetime.now().year

def save_to_audio(text: str) -> None:
    """Converts text to an audio file using Google Text-to-Speech (gTTS)."""
    try:
        tts: gTTS = gTTS(text=text, lang="en")
        tts.save("output.mp3")
        app_logger.log_info("Response converted to audio", level="INFO")
    except Exception as e:
        app_logger.log_error(f"Error converting response to audio: {e}")
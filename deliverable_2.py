from pprint import pprint

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scholarly import scholarly
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


class URLValidator:
    """
    A production-ready URL validation class that evaluates the credibility of a webpage
    using multiple factors: domain trust, content relevance, fact-checking, bias detection, and citations.
    """

    def __init__(self):
        # Load models once to avoid redundant API calls
        self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.fake_news_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

    def fetch_page_content(self, url: str) -> str:
        """ Fetches and extracts text content from the given URL. """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = " ".join([p.text.strip() for p in soup.find_all("p") if p.text.strip()])
            return content if content else ""
        except requests.RequestException:
            return ""

    def get_domain_trust(self, url: str, content: str) -> int:
        """ Computes the domain trust score based on available data sources. """

        # 🔹 Highly Trusted (90-100) - Government, Health, Science, Top Universities
        highly_trusted = {
            # Government and Health Organizations
            "mayoclinic.org": 95, "nih.gov": 95, "fda.gov": 95, "who.int": 95, "cdc.gov": 95,
            "un.org": 95, "nasa.gov": 95, "noaa.gov": 95, "worldbank.org": 95,

            # Top Academic and Research Institutions
            "nature.com": 90, "nejm.org": 90, "jamanetwork.com": 90, "bmj.com": 90, "thelancet.com": 90,
            "mit.edu": 95, "harvard.edu": 95, "stanford.edu": 95, "ox.ac.uk": 95, "cam.ac.uk": 95,
            "berkeley.edu": 90, "princeton.edu": 90, "yale.edu": 90, "columbia.edu": 90,
            "sciencemag.org": 90, "pnas.org": 90, "ieee.org": 90, "researchgate.net": 90,

            # Credible Fact-Checking & Policy
            "factcheck.org": 95, "snopes.com": 90, "politifact.com": 90, "fullfact.org": 90,
            "reuters.com": 95, "apnews.com": 95, "npr.org": 90
        }

        # 🔹 Trusted (70-89) - Established News, Science, and Tech Publications
        trusted = {
            # Major News Outlets
            "bbc.com": 85, "nytimes.com": 85, "theguardian.com": 85, "forbes.com": 80,
            "bloomberg.com": 85, "cnbc.com": 85, "economist.com": 85, "wsj.com": 85, "time.com": 80,
            "sciencedaily.com": 80, "wired.com": 80, "technologyreview.com": 80,

            # Business and Finance
            "marketwatch.com": 80, "fool.com": 75, "investopedia.com": 75, "sec.gov": 90,

            # Science & Technology
            "arxiv.org": 80, "psychologytoday.com": 75, "scirp.org": 70, "nature.scienceopen.com": 80,

            # Popular Educational Platforms
            "khanacademy.org": 85, "coursera.org": 80, "udacity.com": 80, "udemy.com": 75
        }

        # 🔹 Moderately Trusted (50-69) - Blogs, Open-Wiki, Crowd-Sourced, Opinionated News
        moderately_trusted = {
            # Open Wiki & Community-Moderated Content
            "wikipedia.org": 70, "wikihow.com": 65, "stackexchange.com": 65, "stackoverflow.com": 65,

            # Business & Tech Blogs
            "medium.com": 60, "substack.com": 60, "blogspot.com": 55, "businessinsider.com": 60,
            "vox.com": 65, "quora.com": 55, "scirp.org": 60,

            # Tech & Entertainment News
            "engadget.com": 65, "gizmodo.com": 65, "slashdot.org": 65, "techcrunch.com": 65,
            "mashable.com": 60, "theverge.com": 65,

            # Lifestyle & Health Blogs
            "webmd.com": 65, "healthline.com": 65, "self.com": 60, "shape.com": 60
        }

        # 🔹 User-Generated / Low Trust (30-49) - Social Media, Community, Unverified Info
        low_trust = {
            # Social Media & Discussion
            "reddit.com": 45, "quora.com": 40, "tiktok.com": 30, "twitter.com": 40,
            "facebook.com": 40, "instagram.com": 35, "pinterest.com": 35,

            # Crowdsourced News & Open Contributions
            "fandom.com": 35, "wattpad.com": 40, "9gag.com": 35, "buzzfeednews.com": 45,

            # Conspiracy / Alternative Medicine
            "gaia.com": 40, "mercola.com": 40, "sott.net": 30
        }

        # 🔹 Disinformation / Very Low Trust (10-29) - Conspiracy, Satire, Pseudoscience
        very_low_trust = {
            # Known Fake News, Conspiracies, Satire
            "infowars.com": 15, "breitbart.com": 20, "theonion.com": 15, "clickhole.com": 15,
            "dailymail.co.uk": 25, "naturalnews.com": 20, "beforeitsnews.com": 20,
            "prisonplanet.com": 15, "sputniknews.com": 25, "rt.com": 25
        }

        # Assign trust scores based on domain
        for domain_dict in [highly_trusted, trusted, moderately_trusted, low_trust, very_low_trust]:
            for domain, score in domain_dict.items():
                if domain in url:
                    return score

        return 35  # Default trust score for unknown sources

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        """ Computes semantic similarity between user query and page content. """
        if not content or not user_query.strip():
            return 0  # Return 0 if query or content is empty

        try:
            # Process only the first 2000 characters of the content for efficiency
            content = content[:2000]

            # Compute cosine similarity
            similarity = util.pytorch_cos_sim(
                self.similarity_model.encode(user_query, normalize_embeddings=True),
                self.similarity_model.encode(content, normalize_embeddings=True)
            ).item()

            # Scale and ensure similarity score is between 0-100
            return max(0, min(int(similarity * 100), 100))

        except Exception as e:
            print(f"Error in compute_similarity_score: {e}")
            return 0  # Return 0 in case of failure


    def check_facts(self, content: str) -> int:
        """ Cross-checks extracted content with Google Fact Check API. """
        if not content:
            return 50  # Default to neutral if content is empty
        api_url = f"https://toolbox.google.com/factcheck/api/v1/claimsearch?query={content[:200]}"
        try:
            response = requests.get(api_url)
            data = response.json()
            return 90 if "claims" in data and data["claims"] else 40
        except:
            return 50  # Default uncertainty score

    def check_google_scholar(self, query: str) -> int:
        """ Checks the number of citations on Google Scholar for a given query. """
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
            driver.get(url)

            # Look for "Cited by" text
            citations = driver.find_elements("xpath", "//a[contains(text(),'Cited by')]")
            citation_count = int(citations[0].text.split("Cited by ")[1]) if citations else 0

            driver.quit()
            # Change here: return a minimum score instead of 0 if no citations found
            return max(35, min(citation_count * 2, 100))  # Normalize and ensure a minimum score
        except Exception as e:
            print(f"Error fetching Google Scholar citations: {e}")
            return 35  # Provide a minimum score in case of an error or no citations found

    def detect_bias(self, content: str) -> int:
        """ Uses NLP sentiment analysis to detect potential bias in content. """
        if not content:
            return 50  # Neutral if no content is found
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        label = sentiment_result["label"]
        if label == "POSITIVE":
            return 90
        elif label == "NEGATIVE":
            return 30
        else:
            return 60  # Neutral tone

    def get_star_rating(self, score: float) -> tuple:
        """ Converts a score (0-100) into a 1-5 star rating. """
        stars = max(1, min(5, round(score / 20)))  # Normalize 100-scale to 5-star scale
        return stars, "⭐" * stars

    def generate_explanation(self, domain_trust, similarity_score, fact_check_score, bias_score, citation_score, final_score) -> str:
        """ Generates a human-readable explanation for the score. """
        reasons = []
        if domain_trust < 50:
            reasons.append("The source has low domain authority.")
        if similarity_score < 50:
            reasons.append("The content is not highly relevant to your query.")
        if fact_check_score < 50:
            reasons.append("Limited fact-checking verification found.")
        if bias_score < 50:
            reasons.append("Potential bias detected in the content.")
        if citation_score < 30:
            reasons.append("Few citations found for this content.")

        return " ".join(reasons) if reasons else "This source is highly credible and relevant."

    def rate_url_validity(self, user_query: str, url: str) -> dict:
        """ Main function to evaluate the validity of a webpage. """
        content = self.fetch_page_content(url)

        domain_trust = self.get_domain_trust(url, content)
        similarity_score = self.compute_similarity_score(user_query, content)
        fact_check_score = self.check_facts(content)
        bias_score = self.detect_bias(content)
        citation_score = self.check_google_scholar(url)

        final_score = (
            (0.4 * domain_trust) +
            (0.4 * similarity_score) +
            (0.2 * fact_check_score) +
            (0.15 * bias_score) +
            (0.2 * citation_score)
        )

        stars, icon = self.get_star_rating(final_score)
        explanation = self.generate_explanation(domain_trust, similarity_score, fact_check_score, bias_score, citation_score, final_score)

        return {
            "raw_score": {
                "Domain Trust": domain_trust,
                "Content Relevance": similarity_score,
                "Fact-Check Score": fact_check_score,
                "Bias Score": bias_score,
                "Citation Score": citation_score,
                "Final Validity Score": final_score
            },
            "stars": {
                "score": stars,
                "icon": icon
            },
            "explanation": explanation
        }


if __name__ == "__main__":

    # # Dictionary of user prompts and URLs
    data_dict = {
        "China unveils 'Monkey King' Mach 4 supersonic drone": "https://www.thesun.co.uk/tech/33443338/china-monkey-king-supersonic-drone/",
        "Musk's xAI unveils Grok-3 AI chatbot to rival ChatGPT": "https://finance.yahoo.com/news/musks-xai-unveils-grok-3-133237034.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAACYG2A4yzrneGIlw-xmTHUPnKTUgnvVaPmNr0WJ5wZhEpMqltR7kkJ-mRVIOoehIC-wtlCADxJty0_kiP07DbkxINdi7EQnELpiohU2OJ516Mkv-6JCwsmT5cLurU0j7ohxf_hBQDi3LRNzLr2R-5-possovPUyQWmqMfKIbLgkQ",
        "Former OpenAI tech chief Murati's AI startup emerges with 20 hires": "https://www.wired.com/story/mira-murati-startup-hire-staff/",
        "Exclusive: Chinese lithium company halts tech exports amid trade tensions": "https://www.aol.com/news/exclusive-chinese-lithium-company-halts-131455172.html",
        "Groups ask US court to reconsider ruling blocking net neutrality rules": "https://www.reuters.com/world/us/groups-ask-us-court-reconsider-ruling-blocking-net-neutrality-rules-2025-02-18/",
        "Tiger Brokers adopts DeepSeek model as Chinese brokerages embrace AI": "https://www.reuters.com/technology/artificial-intelligence/tiger-brokers-adopts-deepseek-model-chinese-brokerages-funds-rush-embrace-ai-2025-02-18/",
        "China's Baidu says DeepSeek success inspired open-source move": "https://www.scmp.com/tech/big-tech/article/3298981/baidu-adopts-deepseek-ai-models-chasing-tencent-race-embrace-hot-start",
        "Taiwan electronics firms plan more Texas investments, industry body says": "https://economictimes.indiatimes.com/tech/technology/taiwan-electronics-firms-plan-more-texas-investments-industry-body-says/articleshow/118362145.cms",
        "EV battery pack developer Ionetic opens UK pilot production plant": "https://www.electrichybridvehicletechnology.com/news/ionetics-new-5m-battery-plant-aims-to-speed-up-ev-development.html",
        "Vietnam paves way for Musk's Starlink amid US tariff threats": "https://en.tempo.co/read/1977099/vietnam-paves-way-for-musks-starlink-seen-as-olive-branch-amid-us-tariff-threats?utm_source=Digital%20Marketing&utm_medium=Babe",
        "Capgemini sales fall less than expected, but soft outlook affects shares": "https://www.reuters.com/technology/capgemini-posts-2024-sales-slightly-above-estimates-2025-02-18/#:~:text=Feb%2018%20(Reuters)%20%2D%20French,fell%207.7%25%20by%200825%20GMT.",
        "OpenAI considers special voting rights to guard against hostile takeovers": "https://fortune.com/2025/02/18/openai-powersnon-profit-board-hostile-takeover-elon-musk/",
        "Samsung Electronics nominates chip executives as new board members": "https://www.reuters.com/technology/samsung-electronics-nominates-chip-execs-new-board-members-2025-02-18/#:~:text=SEOUL%2C%20Feb%2018%20(Reuters),in%20its%20struggling%20semiconductor%20business.",
        "South Korea aims to secure 10,000 GPUs for national AI computing center": "https://www.yahoo.com/news/south-korea-aims-secure-10-064255061.html",
        "Philippines reports foreign cyber intrusions targeting intelligence data": "https://go.flashpoint.io/2024-global-threat-intelligence-report?utm_campaign=Resource_RP_GTI_2024&utm_source=google&utm_medium=paid-search&utm_term=cyber%20threat%20intelligence%20report&sfcampaign_id=701Rc000008junpIAA&gad_source=1&gclid=Cj0KCQiA_NC9BhCkARIsABSnSTZ5fOs-8aGjeP_DmmIecxhNbC_WRW9DBCsF2VPu0XVzHZsqsDfOFM0aAs_2EALw_wcB",
        "New downloads of DeepSeek suspended in South Korea over data protection": "https://apnews.com/article/south-korea-deepseek-app-downloads-privacy-concerns-ai-20950f357276b9bb8f2a70a4b3c03e96",
        "Nvidia prepares Jetson Thor computers for humanoid robots in 2025": "https://interestingengineering.com/innovation/nvidia-plans-robotic-domination-2025",
        "Google advances AI technology with Gemini 2.0 amid antitrust challenges": "https://www.investors.com/news/technology/google-stock-googl-buy-now-alphabet-stock-february-2025/",
        "China reveals 'White Emperor' supersonic fighter jet capable of space-based weapon deployment": "https://nationaldefence.in/news/china-stuns-the-world-with-successful-test-flight-of-its-sixth-gen-fighter-jet-bai-huangdi-the-white-emperor/#:~:text=Testing%20a%20sixth%20generation%20fighter,White%20Emperor%E2%80%9D%20on%20social%20media.",
        "New touch technologies make smartphones accessible to the blind": "https://www.wsj.com/tech/personal-tech/blind-tactile-technology-devices-apple-dot-newhaptics-122cabe5",
        "Advancements in AI: Generative tech, robots, and emerging risks in 2025": "https://www.technewsworld.com/story/ai-in-2025-generative-tech-robots-and-emerging-risks-179587.html",
        "Lyft plans to introduce robotaxis in Dallas by 2026": "https://www.pcmag.com/news/lyft-aims-to-launch-robotaxis-in-dallas-by-2026",
        "Processor wars: How Qualcomm lost its early lead in AI chips": "https://www.technewsworld.com/story/processor-wars-how-qualcomm-lost-its-early-lead-179578.html",
        "Lenovo's ThinkPad X1 Carbon challenges MacBook Pro dominance": "https://www.technewsworld.com/story/lenovos-thinkpad-x1-carbon-has-me-rethinking-my-macbook-pro-179565.html",
        "Web raiders unleash global brute force attacks from 2.8M IP addresses": "https://www.technewsworld.com/story/web-raiders-unleash-global-brute-force-attacks-from-2-8m-ip-addresses-179589.html",
        "AI 'hallucinations' in court papers pose challenges for lawyers": "https://hai.stanford.edu/news/ai-trial-legal-models-hallucinate-1-out-6-or-more-benchmarking-queries",
        "The young engineers aiding Elon Musk's government takeover": "https://www.wired.com/story/elon-musk-government-young-engineers/",
        "Robots recovering dumped explosives from the Baltic Sea": "https://hakaimagazine.com/features/the-big-baltic-bomb-cleanup/",
        "DOGE staff question 'resign' email as new HR chief dodges answers": "https://www.desmoinesregister.com/story/news/local/west-des-moines/2016/11/15/west-des-moines-settles-sex-discrimination-lawsuit-for-nearly-2-million-dollars/93817208/",
        "The Elektron Digitone II: A modern classic in music production": "https://www.wired.com/review/elektron-digitone-ii/",
        "The best hearing aids of 2025, reviewed by experts": "https://www.cnet.com/health/medical/best-over-the-counter-hearing-aids/",
        "Our favorite digital notebooks and smart pens": "https://www.zdnet.com/article/best-smart-notebook/"}



    # Initialize the validator
    validator = URLValidator()

    # Prepare the data for Excel
    data_list = []
    for user_prompt, url_to_check in data_dict.items():
        # Run the validation
        result = validator.rate_url_validity(user_prompt, url_to_check)

        # Extract the function rating (final validity score converted to a 1-5 scale)
        func_rating = result["stars"]["score"]

        print(f"User Prompt: {user_prompt}/nURL: {url_to_check}/nRating: {func_rating}")

        # pprint(result)

        # Append to data list
        data_list.append([user_prompt, url_to_check, func_rating, ""])

    # Create a DataFrame
    df = pd.DataFrame(data_list, columns=["user_prompt", "url_to_check", "func_rating", "custom_rating"])

    # Save to Excel
    file_path = "url_validation_results.csv"
    df.to_csv(file_path, index=False)

    # # Instantiate the URLValidator class
    # validator = URLValidator()
    #
    # # Define user prompt and URL
    # user_prompt = "I have just been on an international flight, can I come back home to hold my 1-month-old newborn?"
    # url_to_check = "https://www.mayoclinic.org/healthy-lifestyle/infant-and-toddler-health/expert-answers/air-travel-with-infant/faq-20058539"
    #
    # # Run the validation
    # result = validator.rate_url_validity(user_prompt, url_to_check)
    #
    # # Print the results
    # import json
    #
    # print(json.dumps(result, indent=2))

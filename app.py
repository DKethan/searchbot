import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any

import streamlit as st

from helper import ChatBot, current_year, save_to_audio, invoke_duckduckgo_news_search

# ============================ FRONT-END SETUP ============================

st.set_page_config(layout="wide")  # Set Streamlit layout to wide mode
st.title("SearchBot 🤖")  # App title

# ============================ SIDEBAR SETTINGS ============================

with st.sidebar:
    with st.expander("📖 Instruction Manual"):
        st.markdown(
            """
            ## 🧠 SearchBot 🤖 - Your AI-Powered Research Assistant
            Welcome to **SearchBot**, an advanced AI assistant that helps you find the latest news, trends, and information 
            across various sources.

            ### 🔹 How to Use:
            1. **📌 Choose Search Source**  
               - Select the type of search (News, Research Papers, Web Articles).
            2. **📊 Choose Number of Results**  
               - Decide how many results you want (1 to 10).
            3. **🌍 Set Location**  
               - Customize search results based on location.  
               *(e.g., "us-en" for USA, "in-en" for India)*
            4. **⏳ Filter by Time**  
               - Search for the most recent news or past articles:  
                 - **Past Day** 🕐 (Breaking News)  
                 - **Past Week** 🗓 (Trending Topics)  
                 - **Past Month** 📅 (Major Stories)  
                 - **Past Year** ���� (Deep Research)  
            5. **💬 Review Search Results & Chat History**  
               - View results in an interactive table.  
               - Chatbot provides summarized responses with references.

            ---

            ### 🔹 Live Examples You Can Try:
            **📰 Find Latest News**
            - *"What are the latest AI breakthroughs?"*
            - *"Recent developments in space exploration."*

            **📖 Research Papers & Analysis**
            - *"Most cited papers on quantum computing."*
            - *"Deep learning advancements in 2024."*

            **🌍 Location-Based Information**
            - *"Tech news in Silicon Valley."*
            - *"Political updates in the UK."*

            **⚡ AI-Powered Chatbot Insights**
            - *"Summarize recent news on cryptocurrency."*
            - *"Give me top AI news from last week with analysis."*

            """
        )

    # User inputs for search customization
    num: int = st.number_input("📊 Number of results", value=7, step=1, min_value=1, max_value=10)
    location: str = st.text_input("🌍 Location (e.g., us-en, in-en)", value="us-en")
    time_filter: str = st.selectbox(
        "⏳ Time filter",
        ["Past Day", "Past Week", "Past Month", "Past Year"],
        index=1
    )

    # Convert time filter to DuckDuckGo-compatible format
    time_mapping: Dict[str, str] = {"Past Day": "d", "Past Week": "w", "Past Month": "m", "Past Year": "y"}
    time_filter = time_mapping[time_filter]

    only_use_chatbot: bool = st.checkbox("💬 Only use chatbot (Disable Search)")

    # Clear chat history button
    if st.button("🧹 Clear Session"):
        st.session_state.messages = []
        st.rerun()

    # Footer with dynamic year
    st.markdown(f"<h6>📅 Copyright © 2010-{current_year()} Present</h6>", unsafe_allow_html=True)

# ============================ CHAT HISTORY SETUP ============================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# Ensure messages are always a list of dictionaries
if not isinstance(st.session_state.messages, list) or not all(isinstance(msg, dict) for msg in st.session_state.messages):
    st.session_state.messages = []

# Display past chat history in Streamlit chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================ CHAT INPUT & PROCESSING ============================

# Process user input in the chatbox
if prompt := st.chat_input("Ask anything!"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # **Initialize ref_table_string to hold search results**
    ref_table_string: str = "**No references found.**"

    try:
        with st.spinner("Searching..."):  # Show loading spinner
            if only_use_chatbot:
                response: str = "<empty>"
            else:
                # **Call async search function using `asyncio.run()`**
                search_results: Dict[str, Any] = asyncio.run(
                    invoke_duckduckgo_news_search(query=prompt, location=location, num=num, time_filter=time_filter)
                )

                if search_results["status"] == "success":
                    md_data: List[Dict[str, Any]] = search_results["results"]
                    response = f"Here are your search results:\n{md_data}"

                    def clean_title(title: str) -> str:
                        """
                        Cleans the title by replacing '|' with '-' to ensure proper formatting.

                        Args:
                            title (str): The original title.

                        Returns:
                            str: The cleaned title with '|' replaced by '-'.
                        """
                        return title.replace("|", " - ").strip()  # Replace '|' with ' - ' and remove leading/trailing spaces

                    def generate_star_rating(rating: str) -> str:
                        """
                        Converts a numeric rating into a star representation (supports half-stars).

                        Args:
                            rating (str): The rating value as a string.

                        Returns:
                            str: A string representation of the rating using stars (⭐) and half-stars (⭐½).
                        """
                        try:
                            rating_float: float = float(rating)  # Convert rating to float
                            full_stars: int = int(rating_float)  # Extract full stars
                            half_star: str = "⭐½" if (rating_float - full_stars) >= 0.5 else ""  # Add half-star if needed
                            return "⭐" * full_stars + half_star  # Construct final star rating
                        except ValueError:
                            return "N/A"  # Fallback for non-numeric ratings

                    # Start building reference table with proper Markdown formatting
                    ref_table_string = "| Num | Title | Rating | Context |\n|---|------|--------|---------|\n"

                    for res in md_data:
                        # **Fix: Clean the title by replacing '|' with '-'**
                        title_cleaned = clean_title(res['title'])

                        # **Ensure the rating is always numeric before converting to stars**
                        raw_rating = str(res.get('rating', 'N/A')).strip()  # Get rating and strip whitespace

                        # Fix: Only convert rating if it’s a valid number
                        if raw_rating.replace('.', '', 1).isdigit():  # Check if it’s a valid float
                            stars = generate_star_rating(raw_rating)
                        else:
                            stars = "N/A"  # If it's text (like "MIT News"), default to "N/A"

                        # **Ensure proper clickable links in the Title column**
                        if res.get('link', '').startswith("http"):  # Ensure link exists and is valid
                            title = f"[{title_cleaned}]({res['link']})"
                        else:
                            title = title_cleaned  # Fallback to text-only title

                        # **Properly format Context column (limit to 100 chars)**
                        context_summary = res.get('summary', '').strip()  # Ensure it's a string and strip spaces
                        summary = context_summary[:100] + "..." if len(context_summary) > 100 else context_summary

                        # **Final row construction**
                        ref_table_string += f"| {res['num']} | {title} | {stars} | {summary} |\n"

            # **Generate chatbot response based on search results or chat history**
            bot = ChatBot()
            bot.history = st.session_state.messages.copy()
            response = bot.generate_response(
                f"""
                User prompt: {prompt}
                Search results: {response}
                Context: {[res['summary'] for res in search_results.get("results", [])]}
                If search results exist, use them for the answer.
                Otherwise, generate a response based on chat history.
                """
            )

    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        response = "We encountered an issue. Please try again later."

    # **Convert response to audio**
    save_to_audio(response)

    # **Display assistant response in chat UI**
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
        st.audio("output.mp3", format="audio/mpeg", loop=True)
        with st.expander("References:", expanded=True):
            st.markdown(ref_table_string, unsafe_allow_html=True)

    # **Update chat history with final response**
    final_response: str = f"{response}\n\n{ref_table_string}"
    st.session_state.messages.append({"role": "assistant", "content": final_response})

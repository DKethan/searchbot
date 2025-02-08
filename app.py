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
    with st.expander("Instruction Manual"):
        st.markdown(
            """
            ## SearchBot 🤖
            This Streamlit app allows you to search anything.
            ### How to Use:
            1. **Source**: Select your preferred search source.
            2. **Number of Results**: Choose how many results to display.
            3. **Location**: Customize search based on location.
            4. **Response**: View the results in a table.
            5. **Chat History**: Review previous conversations.
            """
        )

    # User inputs for search customization
    num: int = st.number_input("Number of results", value=7, step=1, min_value=1)
    location: str = st.text_input("Location (e.g., us-en, in-en)", value="us-en")
    time_filter: str = st.selectbox("Time filter", ["Past Day", "Past Week", "Past Month", "Past Year"], index=1)

    # Convert time filter to DuckDuckGo-compatible format
    time_mapping: Dict[str, str] = {"Past Day": "d", "Past Week": "w", "Past Month": "m", "Past Year": "y"}
    time_filter = time_mapping[time_filter]

    only_use_chatbot: bool = st.checkbox("Only use chatbot.")  # Option to disable search and use only chatbot

    # Clear chat history button
    if st.button("Clear Session"):
        st.session_state.messages = []
        st.rerun()

    # Footer with dynamic year
    st.markdown(f"<h6>Copyright © 2010-{current_year()} Present</h6>", unsafe_allow_html=True)

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

                    # **Format references as a Markdown Table (Clickable Title, Stars, Context)**
                    ref_table_string = "| Num | Title | Rating | Context |\n|---|------|--------|---------|\n"

                    for res in md_data:
                        # Convert rating number to star symbols (e.g., "⭐⭐⭐")
                        stars: str = "⭐" * int(res['rating']) if res['rating'].isdigit() else "N/A"

                        # Limit summary to 100 characters for brevity
                        summary: str = res['summary'][:100] + "..." if len(res['summary']) > 100 else res['summary']

                        ref_table_string += f"| {res['num']} | [{res['title']}]({res['link']}) | {stars} | {summary} |\n"

                else:
                    response = "No search results found."
                    ref_table_string = "**No references found.**"

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

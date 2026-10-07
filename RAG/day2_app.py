"""
Day 2: A real AI chat interface powered by the free Gemini API.

Run:
streamlit run day2_app.py

Set GEMINI_API_KEY in .streamlit/secrets.toml (or as an environment
variable) before using the chat. Get a free key at
https://aistudio.google.com/apikey  — no credit card required.

Install the SDK with:
pip install google-genai
"""

import os

import streamlit as st
from google import genai


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SelfAI",
    page_icon="✨",
    layout="wide",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background: #000000;
        }

        .hero {
            padding: 2.8rem 1.5rem 2.2rem;
            text-align: center;
            background: linear-gradient(
                125deg,
                #172554,
                #4f46e5 58%,
                #7c3aed
            );
            border-radius: 20px;
            color: white;
            margin-bottom: 1.6rem;
        }

        .hero h1 {
            font-size: 2.6rem;
            margin: 0;
        }

        .hero p {
            font-size: 1.1rem;
            opacity: .88;
            margin: .7rem 0 0;
        }

        .card {
            background: white;
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
        }

        .tag {
            color: #4338ca;
            background: #eef2ff;
            padding: .25rem .6rem;
            border-radius: 20px;
            font-size: .85rem;
            font-weight: 600;
        }
    </style>

    <section class="hero">
        <h1>✨ My Self AI Career Assistant</h1>
        <p>Ask questions, learn faster, and prepare with me.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Get API key
# --------------------------------------------------

def get_api_key():
    """Read API key from Streamlit secrets or environment variable."""

    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GEMINI_API_KEY")


# --------------------------------------------------
# Gemini API call
# --------------------------------------------------

SYSTEM_INSTRUCTIONS = (
    "You are SelfAI, a friendly career and interview-preparation "
    "assistant. Give practical, concise answers. Do not invent "
    "personal experience or facts."
)

# Free-tier model. Swap for "gemini-2.5-flash-lite" if you hit rate
# limits, or "gemini-2.5-pro" for higher-quality (lower free quota) answers.
MODEL_NAME = "gemini-3-flash-preview"


def answer_with_ai(messages):
    """Send conversation to Gemini and return generated text."""

    api_key = get_api_key()

    if not api_key:
        return None, (
            "No API key found. Add GEMINI_API_KEY "
            "to .streamlit/secrets.toml. Get a free key at "
            "https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    # Gemini uses role "model" instead of "assistant", and expects
    # each turn as {"role": ..., "parts": [{"text": ...}]}.
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
    ]

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"system_instruction": SYSTEM_INSTRUCTIONS},
        )
    except Exception as error:  # noqa: BLE001 - surfaced to the user below
        return None, f"The AI request could not finish: {error}"

    return response.text, None


# --------------------------------------------------
# Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I am your SelfAI assistant. "
                "Ask me about interview preparation."
            ),
        }
    ]


# --------------------------------------------------
# Layout
# --------------------------------------------------

left, right = st.columns([2, 1], gap="large")


# --------------------------------------------------
# Chat section
# --------------------------------------------------

with left:

    st.subheader("Chat with your assistant")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input(
        "Ask a question about interviews..."
    )

    if prompt:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("SelfAI is thinking..."):

                reply, error = answer_with_ai(
                    st.session_state.messages
                )

                if error:
                    st.error(error)

                else:
                    st.markdown(reply)

                    # Store assistant response
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                        }
                    )


# --------------------------------------------------
# Right-side information card
# --------------------------------------------------

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="tag">DAY 2</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### What changed")

    st.markdown(
        """
        - Google Gen AI SDK (`google-genai`)
        - Free Gemini API key (no card needed)
        - Real model response
        - Conversation history
        """
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    # Clear conversation button
    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I am your SelfAI assistant. "
                    "Ask me about interview preparation."
                ),
            }
        ]

        st.rerun()


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Day 2 • Real generative AI with Google's free Gemini API"
)
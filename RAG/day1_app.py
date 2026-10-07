"""Day 1: UI-only AI assistant.

Run with: streamlit run day1_app.py
This file teaches Streamlit structure and CSS before adding RAG.
"""

import streamlit as st


st.set_page_config(page_title="Self AI", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background: #f6f8fc; }
        .hero {
            padding: 2.8rem 1.5rem 2.2rem; text-align: center;
            background: linear-gradient(125deg, #172554, #4f46e5 58%, #7c3aed);
            border-radius: 20px; color: white; margin-bottom: 1.6rem;
        }
        .hero h1 { font-size: 2.6rem; margin: 0; }
        .hero p { font-size: 1.1rem; opacity: .88; margin: .7rem 0 0; }
        .card {
            background: white; padding: 1.2rem; border-radius: 16px;
            border: 1px solid #e5e7eb; min-height: 120px;
        }
        .tag {
            color: #4338ca; background: #eef2ff; padding: .25rem .6rem;
            border-radius: 20px; font-size: .85rem; font-weight: 600;
        }
    </style>
    <section class="hero">
        <h1>✦ My Self AI Career Assistant</h1>
        <p>Ask questions, learn faster, and prepare with me.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am your Self AI assistant. Ask me about interview preparation."}
    ]

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Chat with your assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask a question about interviews...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Day 1 uses a predictable placeholder. Day 2 will replace this with an LLM API call.
        reply = (
            f"Great question: **{prompt}**\n\n"
            "This is the Day 1 UI response. Next, we will connect it to a real AI model."
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="tag">DAY 1</span>', unsafe_allow_html=True)
    st.markdown("### What you built")
    st.markdown("- Page layout with columns\n- Custom CSS styling\n- Chat interface\n- Session-based message history")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I am your CareerAI assistant. Ask me about interview preparation."}
        ]
        st.rerun()

st.caption("Day 1 of your RAG + Agentic AI project • Built with Python, Streamlit, HTML and CSS")

"""
OpenAI ChatKit integration for Social Media Dashboard
Using streamlit-openai package: https://github.com/sbslee/streamlit-openai
"""

import streamlit as st

try:
    import streamlit_openai
    STREAMLIT_OPENAI_AVAILABLE = True
except ImportError:
    STREAMLIT_OPENAI_AVAILABLE = False


def render():
    """
    Render OpenAI ChatKit component in the Social Media Dashboard.
    Uses streamlit-openai package for a better chat experience.
    """
    if not STREAMLIT_OPENAI_AVAILABLE:
        st.error("❌ `streamlit-openai` package is not installed. Please install it with: `pip install streamlit-openai`")
        st.info("The chat interface requires the streamlit-openai package to function.")
        return
    
    st.markdown("### 💬 AI Assistant")
    st.markdown("Ask questions about your social media performance and insights.")
    
    # Initialize chat if not already in session state
    if "chat" not in st.session_state:
        st.session_state.chat = streamlit_openai.Chat()
    
    # Run the chat interface
    st.session_state.chat.run()

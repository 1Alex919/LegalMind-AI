"""Main Streamlit application."""

import sys
from pathlib import Path

# Add project root to Python path so that src/config/ui packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="LegalMind AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .risk-critical { border-left: 4px solid #dc3545; padding-left: 1rem; }
    .risk-high { border-left: 4px solid #fd7e14; padding-left: 1rem; }
    .risk-medium { border-left: 4px solid #ffc107; padding-left: 1rem; }
    .risk-low { border-left: 4px solid #28a745; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("LegalMind AI")
    st.caption("AI-powered legal document analysis")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Upload", "Risk Analysis", "Q&A Chat", "Summary"],
        label_visibility="collapsed",
    )

    st.divider()

    # Document info
    doc = st.session_state.get("document")
    if doc:
        st.markdown("**Current Document**")
        st.text(doc["filename"])
        st.text(f"{doc['total_pages']} pages | {doc['chunks_stored']} chunks")
    else:
        st.info("No document loaded")

    st.divider()
    st.caption("Built with LangChain, LangGraph, ChromaDB, and OpenAI")

# Main content
st.title("LegalMind AI")

if page == "Upload":
    from ui.components.upload import render_upload

    render_upload()

elif page == "Risk Analysis":
    from ui.components.risk_visualizer import render_risk_analysis

    render_risk_analysis()

elif page == "Q&A Chat":
    from ui.components.chat import render_chat

    render_chat()

elif page == "Summary":
    from src.agents.orchestrator import run

    doc = st.session_state.get("document")
    if not doc:
        st.info("Upload a document first to generate a summary.")
    else:
        st.subheader("Contract Summary")

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                try:
                    state = run(task_type="summary", context=doc["full_text"][:8000])
                    result = state.get("result", {})
                    st.session_state["summary"] = result
                except Exception as e:
                    st.error(f"Summary failed: {e}")

        summary_data = st.session_state.get("summary")
        if summary_data:
            st.markdown(f"**Contract Type:** {summary_data.get('contract_type', 'Unknown')}")
            st.markdown(f"**Parties:** {', '.join(summary_data.get('parties', []))}")
            st.divider()
            st.markdown(summary_data.get("summary", ""))
            st.divider()
            st.markdown("**Key Points:**")
            for point in summary_data.get("key_points", []):
                st.markdown(f"- {point}")

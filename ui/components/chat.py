"""Q&A chat interface for Streamlit."""

import streamlit as st

from src.agents.orchestrator import run


def render_chat() -> None:
    """Render the Q&A chat interface."""
    doc = st.session_state.get("document")
    if not doc:
        st.info("Upload a document first to ask questions.")
        return

    st.subheader("Ask Questions")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(
                            f"Page {src.get('page', '?')} | "
                            f"Relevance: {src.get('relevance_score', 0):.2f}"
                        )
                        st.text(src.get("text", "")[:300])
            if msg.get("confidence") is not None:
                st.progress(msg["confidence"], text=f"Confidence: {msg['confidence']:.0%}")

    # Chat input
    question = st.chat_input("Ask about the contract...")

    if question:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    state = run(task_type="qa", question=question)
                    result = state.get("result", {})

                    answer = result.get("answer", "I couldn't generate an answer.")
                    confidence = result.get("confidence", 0.0)
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.caption(
                                    f"Page {src.get('page', '?')} | "
                                    f"Relevance: {src.get('relevance_score', 0):.2f}"
                                )
                                st.text(src.get("text", "")[:300])

                    st.progress(confidence, text=f"Confidence: {confidence:.0%}")

                    # Save to history
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "confidence": confidence,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

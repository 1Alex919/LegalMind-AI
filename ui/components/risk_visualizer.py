"""Risk visualization charts and highlighting for Streamlit."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.orchestrator import run


SEVERITY_COLORS = {
    "critical": "#dc3545",
    "high": "#fd7e14",
    "medium": "#ffc107",
    "low": "#28a745",
}

SEVERITY_ORDER = ["critical", "high", "medium", "low"]


def render_risk_analysis() -> None:
    """Render risk analysis with visualization."""
    doc = st.session_state.get("document")
    if not doc:
        st.info("Upload a document first to analyze risks.")
        return

    st.subheader("Risk Analysis")

    if st.button("Analyze Risks", type="primary"):
        with st.spinner("Analyzing contract for risks..."):
            try:
                state = run(task_type="risk", context=doc["full_text"][:8000])
                result = state.get("result", {})
                st.session_state["risks"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    risks_data = st.session_state.get("risks")
    if not risks_data:
        return

    risks = risks_data.get("risks", [])
    summary = risks_data.get("summary", "")

    if summary:
        st.info(summary)

    if not risks:
        st.success("No significant risks identified.")
        return

    # Severity distribution chart
    severity_counts = {}
    for r in risks:
        sev = r.get("severity", "medium").lower()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    ordered_data = [(s, severity_counts.get(s, 0)) for s in SEVERITY_ORDER if s in severity_counts]
    if ordered_data:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[d[0].title() for d in ordered_data],
                    y=[d[1] for d in ordered_data],
                    marker_color=[SEVERITY_COLORS.get(d[0], "#6c757d") for d in ordered_data],
                )
            ]
        )
        fig.update_layout(
            title="Risk Severity Distribution",
            xaxis_title="Severity",
            yaxis_title="Count",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk cards
    for i, risk in enumerate(risks):
        severity = risk.get("severity", "medium").lower()
        color = SEVERITY_COLORS.get(severity, "#6c757d")

        with st.expander(
            f"{'🔴' if severity == 'critical' else '🟠' if severity == 'high' else '🟡' if severity == 'medium' else '🟢'} "
            f"[{severity.upper()}] {risk.get('risk_type', 'Unknown')}",
            expanded=(severity in ("critical", "high")),
        ):
            st.markdown(f"**Clause:** _{risk.get('clause_text', 'N/A')}_")
            st.markdown(f"**Explanation:** {risk.get('explanation', 'N/A')}")
            st.markdown(f"**Recommendation:** {risk.get('recommendation', 'N/A')}")
            if risk.get("page"):
                st.caption(f"Page {risk['page']}")

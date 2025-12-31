import gradio as gr
from reasoning.graph import app
from reasoning.graph_state import GraphState


# ----------------------------------------
# Gradio Chat Function
# ----------------------------------------
def chat(query):
    final_state = app.invoke(
        GraphState(query=query)
    )

    # Handle BOTH dict and GraphState outputs safely
    if isinstance(final_state, dict):
        answer = final_state.get("answer", "No answer generated.")
    else:
        answer = getattr(final_state, "answer", None) or "No answer generated."

    answer_clean = answer.strip().lower()

    #OUT-OF-SCOPE → NO SOURCES
    if answer_clean.startswith("i could not find relevant information"):
        sources_text = "No sources"
        return answer, sources_text

    # ✅ IN-SCOPE → extract ONLY cited Primary Source from answer
    sources = []
    if "Primary Source:" in answer:
        source_line = answer.split("Primary Source:", 1)[1].strip()
        sources.append(f"- {source_line}")

    sources_text = "\n".join(sources) if sources else "No sources"

    return answer, sources_text


# ----------------------------------------
# Gradio UI Layout
# ----------------------------------------
with gr.Blocks(title="Customer Support Intelligence System") as demo:
    gr.Markdown("## 📞 Customer Support Intelligence System")
    gr.Markdown(
        "Ask questions about **FAQs, SLAs, Internal Policies, and Escalation Workflows**.\n"
        "Responses are grounded strictly in internal documents."
    )

    query_input = gr.Textbox(
        label="Customer Question",
        placeholder="e.g., Can I order from multiple restaurants at once?"
    )

    answer_output = gr.Textbox(
        label="Answer",
        lines=6
    )

    source_output = gr.Textbox(
        label="Sources",
        lines=4
    )

    submit_btn = gr.Button("Ask")

    submit_btn.click(
        fn=chat,
        inputs=query_input,
        outputs=[answer_output, source_output]
    )

demo.launch()

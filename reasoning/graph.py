from langgraph.graph import StateGraph
from reasoning.graph_state import GraphState
from reasoning.nodes import *

graph = StateGraph(GraphState)

graph.add_node("intent", intent_detection_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("grounding", grounding_node)
graph.add_node("safety", safety_guardrail_node)
graph.add_node("response", response_node)
graph.add_node("audit", audit_log_node)

graph.set_entry_point("intent")

graph.add_edge("intent", "retrieval")
graph.add_edge("retrieval", "grounding")
graph.add_edge("grounding", "safety")
graph.add_edge("safety", "response")
graph.add_edge("response", "audit")   # 👈 Task 5

app = graph.compile()

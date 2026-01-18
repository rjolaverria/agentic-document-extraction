import sys
from uuid import uuid4

# Import compatibility shim BEFORE PaddleOCR
import shims.langchain_compat as langchain_compat  # noqa: F401

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.messages.utils import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from ocr.text_extraction import get_ordered_text

model = ChatOpenAI(model="gpt-5.2", temperature=0)


def run(image_path: str):
    request_id = str(uuid4())
    ordered_text = get_ordered_text(request_id, image_path)
    ordered_text_str = "\n".join(
        [f"[{item['position']}] {item['text']}" for item in ordered_text]
    )

    system_prompt = f"""
You are a Document Extractor Agent. 
You analyze documents and extract relevant information based on the provided context. When instructed, you should provide concise and accurate responses.

## Document Text (in reading order)
The following text was extracted using OCR and ordered using LayoutLM.

{ordered_text_str}
"""
    agent: CompiledStateGraph = create_agent(
        model=model,
        system_prompt=system_prompt,
    )

    messages: list[AnyMessage] = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        messages.append(HumanMessage(user_input))
        response = agent.invoke({"messages": messages})
        messages = response["messages"]
        print("Assistant:", messages[-1].content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path/to/image>")
        sys.exit(1)

    image_path = sys.argv[1]
    run(image_path)

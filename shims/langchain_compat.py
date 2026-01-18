# mypy: ignore-errors
"""
Compatibility shim for PaddleOCR's outdated langchain imports.
Import this module before importing PaddleOCR to fix module compatibility issues.
"""

import sys
from types import ModuleType

# Only create the missing langchain.docstore module structure if it doesn't exist
if "langchain.docstore" not in sys.modules:
    docstore = ModuleType("langchain.docstore")
    sys.modules["langchain.docstore"] = docstore

if "langchain.docstore.document" not in sys.modules:
    document = ModuleType("langchain.docstore.document")
    sys.modules["langchain.docstore.document"] = document

    # Import the actual Document class from the new location
    from langchain_community.docstore.document import Document  # type: ignore[attr-defined]

    document.Document = Document  # type: ignore[attr-defined]

# Create the missing langchain.text_splitter module if it doesn't exist
if "langchain.text_splitter" not in sys.modules:
    text_splitter = ModuleType("langchain.text_splitter")
    sys.modules["langchain.text_splitter"] = text_splitter

    # Import the actual RecursiveCharacterTextSplitter from the new location
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[attr-defined]

    text_splitter.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter  # type: ignore[attr-defined]

import os
import re
import sys
from typing import List, Dict, Optional

# NOTE: This is a simplified history parser for the one-off script execution model.
# It reads the *entire* history file on each run and does *not* implement
# summarization like Aider's full implementation. For long conversations,
# this approach may lead to exceeding the LLM's context window limits.
# Future work should integrate a proper in-memory history and summarization
# mechanism if the main workflow becomes persistent.

def parse_history_markdown(history_file_path: Optional[str]) -> List[Dict[str, str]]:
    """
    Parses the chat history markdown file into a list of message dictionaries.

    Args:
        history_file_path: Path to the markdown history file.

    Returns:
        A list of dictionaries, e.g., [{"role": "user", "content": "..."}, ...].
        Returns an empty list if the file doesn't exist or is empty.
    """
    if not history_file_path or not os.path.exists(history_file_path):
        return []

    try:
        with open(history_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read history file {history_file_path}: {e}", file=sys.stderr)
        return []

    if not content.strip():
        return []

    messages = []
    # Regex to find role headers and capture the content following them
    # It looks for #### Role @ Timestamp headers and captures everything until the next header or end of file
    pattern = re.compile(r"#### (User|Assistant) @ .*?\n(.*?)(?=#### (?:User|Assistant) @ |\Z)", re.DOTALL | re.MULTILINE)

    for match in pattern.finditer(content):
        role = match.group(1).lower()
        message_content = match.group(2).strip()

        if role in ["user", "assistant"] and message_content:
            messages.append({"role": role, "content": message_content})
        elif role not in ["user", "assistant"]:
             print(f"Warning: Skipping unrecognized role '{role}' in history file.", file=sys.stderr)


    # Basic validation: Check if parsing resulted in any messages if content was present
    if content.strip() and not messages:
         print(f"Warning: History file {history_file_path} has content but parsing yielded no messages. Check format.", file=sys.stderr)


    return messages

# Example usage (for testing the parser directly)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Parsing history file: {test_file}")
        parsed_history = parse_history_markdown(test_file)
        import json
        print(json.dumps(parsed_history, indent=2))
    else:
        print("Usage: python history_parser.py <path_to_history_file.md>")

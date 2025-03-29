#!/usr/bin/env python

"""
Main workflow script for the simplified Aider clone.

Based on code from the Aider project: https://github.com/paul-gauthier/aider

This script orchestrates the complete workflow for interacting with an LLM to assist with
code development. It handles:

1. Taking user input and configuration
2. Building comprehensive prompts using prompt_builder.py
3. Sending prompts to the LLM via llm.py
4. Streaming the LLM response to stdout
5. Logging the interaction to a history markdown file

The workflow follows Aider's approach of maintaining context through:
- Repository mapping and file context
- Chat history persistence
- Structured prompt building
- Consistent message formatting

Key Features:
- Integrates with prompt_builder.py for prompt construction
- Uses llm.py for LLM communication
- Maintains chat history in markdown format
- Supports streaming responses
- Handles error cases gracefully
- Provides verbose output for debugging

Usage:
  python main_workflow.py <user_prompt> --dir <project_root> [options]

Install dependencies:
  pip install tiktoken litellm
"""

import argparse
import json
import os
import re
# Removed subprocess import
import sys
from datetime import datetime
# Assuming llm.py and prompt_builder.py are in the same directory or Python path
from llm import LLMClient
from prompt_builder import PromptBuilder # Import the new class


def append_to_history(history_file: str, user_prompt: str, assistant_response: str):
    """Appends the user prompt and assistant response to the history file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f"#### User @ {timestamp}\n")
            f.write(f"{user_prompt.strip()}\n\n")
            f.write(f"#### Assistant @ {timestamp}\n")
            f.write(f"{assistant_response.strip()}\n\n")
    except IOError as e:
        print(f"Warning: Could not write to history file {history_file}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: An unexpected error occurred writing to history file: {e}", file=sys.stderr)

# Removed the run_prompt_builder function as we now import directly


def main():
    parser = argparse.ArgumentParser(description="Simplified Aider workflow.")
    parser.add_argument("user_prompt", help="The user's request/prompt.")
    parser.add_argument("--dir", required=True, help="Root directory of the project.")
    parser.add_argument("--chat-files", nargs='*', default=[], help="Relative paths of files in the chat.")
    parser.add_argument("--read-only-files", nargs='*', default=[], help="Relative paths of read-only files.")
    parser.add_argument("--model", default=os.getenv("AIDER_MODEL", "gpt-4o-mini"), help="LLM model name.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key for the LLM service.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE"), help="Base URL for custom LLM endpoints.")
    parser.add_argument("--history-file", default=".aidermacs_history.md", help="Path to the chat history markdown file.")
    # Removed --prompt-builder-script argument
    parser.add_argument("--map-tokens", type=int, default=4096, help="Max tokens for repo map (passed to PromptBuilder).")
    parser.add_argument("--tokenizer", default="cl100k_base", help="Tokenizer name (passed to PromptBuilder).")
    parser.add_argument("--no-shell", action="store_true", help="Disable shell command suggestions in the prompt (passed to PromptBuilder).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--print-prompt", action="store_true", help="Print the full prompt messages to stderr before sending.")

    args = parser.parse_args()

    # --- Pre-process: Find and add @-mentioned files ---
    mentioned_in_prompt = set()
    pattern = r'@(\S+)' # Find @ followed by non-whitespace characters
    matches = re.findall(pattern, args.user_prompt)
    if matches:
        if args.verbose:
            print(f"Found potential @-mentions: {matches}", file=sys.stderr)
        for potential_file in matches:
            # Strip trailing punctuation that might be attached
            potential_file = potential_file.rstrip('.,;:!?')
            abs_path = os.path.abspath(os.path.join(args.dir, potential_file))
            if os.path.isfile(abs_path):
                # Use the relative path as provided in the mention
                mentioned_in_prompt.add(potential_file)
                if args.verbose:
                    print(f"  Validated and adding to chat_files: {potential_file}", file=sys.stderr)
            elif args.verbose:
                print(f"  Ignoring mention '{potential_file}': File not found or not a file at {abs_path}", file=sys.stderr)

    # Combine CLI args with prompt mentions, ensuring uniqueness
    original_chat_files = set(args.chat_files)
    updated_chat_files = sorted(list(original_chat_files.union(mentioned_in_prompt)))

    if args.verbose and updated_chat_files != args.chat_files:
        print(f"Updated chat_files list: {updated_chat_files}", file=sys.stderr)
    args.chat_files = updated_chat_files # Update args object

    # --- 1. Build the Prompt using imported PromptBuilder ---
    if args.verbose:
        print("\n--- Building prompt using PromptBuilder ---", file=sys.stderr)

    try:
        builder = PromptBuilder(
            root_dir=args.dir,
            user_message=args.user_prompt,
            chat_files=args.chat_files,
            read_only_files=args.read_only_files,
            map_tokens=args.map_tokens,
            tokenizer=args.tokenizer,
            verbose=args.verbose,
            no_shell=args.no_shell,
            history_file=args.history_file, # Pass history file path
            # Assuming default fences '```' are okay, add args if needed
        )
        messages = builder.build_prompt_messages()

        if args.verbose:
             print("--- PromptBuilder output (messages) ---", file=sys.stderr)
             # Avoid printing full base64 images if any were included
             printable_messages = []
             for msg in messages:
                 if isinstance(msg.get("content"), list): # Handle image messages
                     new_content = []
                     for item in msg["content"]:
                         if isinstance(item, dict) and item.get("type") == "image_url":
                              img_url = item.get("image_url", {}).get("url", "")
                              if isinstance(img_url, str) and img_url.startswith("data:"):
                                  new_content.append({"type": "image_url", "image_url": {"url": img_url[:50] + "..."}})
                              else:
                                  new_content.append(item)
                         else:
                             new_content.append(item)
                     printable_messages.append({"role": msg["role"], "content": new_content})
                 else:
                     printable_messages.append(msg)
             print(json.dumps(printable_messages, indent=2), file=sys.stderr)
             print("--- End PromptBuilder output ---", file=sys.stderr)

    except Exception as e:
        print(f"Error during prompt building: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- Optional: Print the full prompt before sending ---
    if args.print_prompt:
        print("\n--- Full Prompt to LLM ---", file=sys.stderr)
        # Use the same printable logic as verbose output for messages
        printable_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), list): # Handle image messages
                new_content = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                         img_url = item.get("image_url", {}).get("url", "")
                         if isinstance(img_url, str) and img_url.startswith("data:"):
                             new_content.append({"type": "image_url", "image_url": {"url": img_url[:50] + "..."}})
                         else:
                             new_content.append(item)
                    else:
                        new_content.append(item)
                printable_messages.append({"role": msg["role"], "content": new_content})
            else:
                printable_messages.append(msg)
        print(json.dumps(printable_messages, indent=2), file=sys.stderr)
        print("--- End Full Prompt ---", file=sys.stderr)


    # --- 2. Interact with LLM ---
    client = LLMClient(
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        verbose=args.verbose,
    )

    print("\nAssistant:") # Header for the output
    full_response = ""
    try:
        # Send the messages generated by prompt_builder directly
        response_stream = client.send(messages, stream=True)
        for chunk in response_stream:
            print(chunk, end="", flush=True)
            full_response += chunk
        print() # Ensure a newline after the stream

    except Exception as e:
        print(f"\nError during LLM communication: {e}", file=sys.stderr)
        # Decide if you want to exit or just log the error
        # For now, we'll log and continue to history writing if possible
        full_response = f"[Error during LLM communication: {e}]"


    # --- 3. Log History ---
    if args.history_file:
        append_to_history(args.history_file, args.user_prompt, full_response)
        if args.verbose:
            print(f"\nInteraction logged to {args.history_file}", file=sys.stderr)


if __name__ == "__main__":
    main()

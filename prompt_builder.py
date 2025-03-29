#!/usr/bin/env python

"""
Standalone script to build prompts using Aider's exact formulation approach.

Based on code from the Aider project: https://github.com/paul-gauthier/aider

This script constructs prompts in the same structured format used by Aider when
communicating with Large Language Models (LLMs). It handles:

1. System prompt setup with platform info and configuration
2. Repository context via repomap.py integration
3. File content inclusion (chat files and read-only files)
4. Message history management
5. Contextual prefixes and assistant responses

The prompt structure follows Aider's ChatChunks format, which organizes messages into:
- System instructions
- Example conversations
- Repository context
- Read-only file references
- Chat file contents
- Current user message
- Reminders/context refreshers

Key Features:
- Integrates with repomap.py for repository context
- Extracts file/identifier mentions from user messages
- Supports multiple fence styles for code blocks
- Handles shell command suggestions
- Maintains Aider's exact message formatting

Usage:
  python prompt_builder.py --dir <project_root> --user-message <message> [options]

Install dependencies:
  pip install tiktoken
"""

import argparse
import json
import locale
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from prompt_templates import RepoPrompts
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import the history parser function
from history_parser import parse_history_markdown
from prompt_templates import RepoPrompts
from repomapper import RepoMapper


# --- Helper Functions ---

def find_src_files(directory):
    """Finds potentially relevant files, mimicking part of RepoMap/BaseCoder logic."""
    src_files = []
    # Basic exclusion list, similar to repomap.py
    exclude_dirs = {'.git', 'node_modules', 'vendor', 'build', 'dist', '__pycache__', '.venv', 'env'}
    exclude_exts = {'.log', '.tmp', '.bak', '.swp', '.pyc'} # Example non-source extensions

    for root, dirs, files in os.walk(directory, topdown=True):
        # Filter excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            file_path = Path(os.path.join(root, file))
            # Basic filtering: Skip common non-source file types if needed
            if file_path.suffix.lower() not in exclude_exts:
                try:
                    # Ensure it's a file and readable (basic check)
                    if file_path.is_file():
                         # Get relative path
                         rel_path = os.path.relpath(file_path, directory)
                         src_files.append(rel_path)
                except OSError:
                    continue # Skip files that cause OS errors (e.g., permission denied)
    return src_files

def get_rel_fname(fname, root):
    """Gets the relative path of fname from the root."""
    try:
        return os.path.relpath(fname, root)
    except ValueError:
        return fname # Handle different drives on Windows

def get_all_relative_files(root_dir, chat_files, read_only_files):
    """Simulates BaseCoder's get_all_relative_files for the standalone script."""
    # In the standalone script, we might not have a full git repo context.
    # We'll rely on finding files within the specified directory.
    all_found = find_src_files(root_dir)
    # Ensure chat_files and read_only_files are also included if they exist
    # (find_src_files might miss them if they are outside typical source dirs)
    existing_files = set(all_found)
    for f in chat_files + read_only_files:
        abs_path = os.path.abspath(os.path.join(root_dir, f))
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            existing_files.add(f)
    return sorted(list(existing_files))


def get_addable_relative_files(root_dir: str, chat_files: List[str], read_only_files: List[str]) -> List[str]:
    """Simulates BaseCoder's get_addable_relative_files."""
    all_files = set(get_all_relative_files(root_dir, chat_files, read_only_files))
    inchat_files = set(chat_files)
    readonly_files = set(read_only_files)
    return sorted(list(all_files - inchat_files - readonly_files))


def get_file_mentions(content: str, root_dir: str, chat_files: List[str], read_only_files: List[str]) -> List[str]:
    """Extracts potential file path mentions from text, adapted from BaseCoder."""
    words = set(word for word in content.split())
    words = set(word.rstrip(",.!;:?") for word in words) # Drop punctuation
    quotes = "".join(['"', "'", "`"])
    words = set(word.strip(quotes) for word in words) # Strip quotes

    addable_rel_fnames = get_addable_relative_files(root_dir, chat_files, read_only_files)

    # Get basenames of files already in chat or read-only to avoid re-suggesting them by basename
    existing_basenames = {os.path.basename(f) for f in chat_files} | \
                         {os.path.basename(f) for f in read_only_files}

    mentioned_rel_fnames = set()
    fname_to_rel_fnames = {}
    for rel_fname in addable_rel_fnames:
        # Skip files that share a basename with files already in chat/read-only
        if os.path.basename(rel_fname) in existing_basenames:
            continue

        # Normalize paths for comparison (e.g., Windows vs Unix separators)
        normalized_rel_fname = rel_fname.replace("\\", "/")
        normalized_words = set(word.replace("\\", "/") for word in words)

        # Direct match of relative path
        if normalized_rel_fname in normalized_words:
            mentioned_rel_fnames.add(rel_fname)

        # Consider basename matches, but only if they look like filenames
        # and don't conflict with existing files
        fname = os.path.basename(rel_fname)
        if "/" in fname or "\\" in fname or "." in fname or "_" in fname or "-" in fname:
            if fname not in fname_to_rel_fnames:
                fname_to_rel_fnames[fname] = []
            fname_to_rel_fnames[fname].append(rel_fname)

    # Add unique basename matches
    for fname, rel_fnames in fname_to_rel_fnames.items():
        if len(rel_fnames) == 1 and fname in words:
            mentioned_rel_fnames.add(rel_fnames[0])

    return list(mentioned_rel_fnames)


def get_ident_mentions(text: str) -> List[str]:
    """Extracts potential identifiers (words) from text, adapted from BaseCoder."""
    # Split on non-alphanumeric characters
    words = set(re.split(r"\W+", text))
    # Filter out short words or purely numeric strings
    return [word for word in words if len(word) >= 3 and not word.isdigit()]


# --- ChatChunks Class (Structure for organizing prompt parts) ---

class ChatChunks:
    """Structures the prompt with clear separation between examples, history, and conversation"""
    def __init__(self):
        self.system = []          # System instructions
        self.examples = []        # Few-shot examples
        self.history = []         # Parsed conversation history (Compromise: read from file)
        self.context = []         # Context added *after* history (repo map, files)
        self.current = []         # Current user message and response

    def all_messages(self):
        """Combine all message chunks in the correct order"""
        messages = []
        messages.extend(self.system)

        # Add few-shot examples with clear separator
        if self.examples:
            messages.extend(self.examples)
            messages.append({
                "role": "user",
                "content": "I switched to a new code base. Please don't consider the above files or try to edit them any longer."
            })
            messages.append({
                "role": "assistant",
                "content": "Ok."
            })

        # Add parsed history messages after examples, before current context
        # NOTE: This loads the *entire* history file content. No summarization.
        # This might exceed token limits for long-running chats.
        messages.extend(self.history)

        # Add current context (repo map, files) after history
        messages.extend(self.context)

        # Add the current user message and final system reminder
        messages.extend(self.current)
        return messages


# --- Platform Info ---

def get_platform_info(repo: bool = True) -> str:
    """Generate platform info matching Aider's format"""
    platform_text = f"- Platform: {platform.platform()}\n"
    shell_var = "COMSPEC" if os.name == "nt" else "SHELL"
    shell_val = os.getenv(shell_var)
    platform_text += f"- Shell: {shell_var}={shell_val}\n"

    try:
        lang = locale.getlocale()[0]
        if lang:
            platform_text += f"- Language: {lang}\n"
    except Exception:
        pass

    dt = datetime.now().astimezone().strftime("%Y-%m-%d")
    platform_text += f"- Current date: {dt}\n"

    if repo:
        platform_text += "- The user is operating inside a git repository\n"

    return platform_text


# --- System Prompt Formatting ---

def fmt_system_prompt(
    prompts: RepoPrompts,
    fence: Tuple[str, str],
    platform_text: str,
    suggest_shell_commands: bool = True
) -> str:
    """Format system prompt using RepoPrompts templates."""
    lazy_prompt = prompts.lazy_prompt # Assuming lazy model is not configurable here
    language = "the same language they are using" # Default language behavior

    if suggest_shell_commands:
        shell_cmd_prompt = prompts.shell_cmd_prompt.format(platform=platform_text)
        shell_cmd_reminder = prompts.shell_cmd_reminder.format(platform=platform_text)
    else:
        shell_cmd_prompt = prompts.no_shell_cmd_prompt.format(platform=platform_text)
        shell_cmd_reminder = prompts.no_shell_cmd_reminder.format(platform=platform_text)

    # Basic fence check for quad backtick reminder (can be enhanced)
    quad_backtick_reminder = (
        "\nIMPORTANT: Use *quadruple* backticks ```` as fences, not triple backticks!\n"
        if fence[0] == "`" * 4 else ""
    )

    # Use the main_system template from RepoPrompts
    system_content = prompts.main_system.format(
        fence=fence,
        quad_backtick_reminder=quad_backtick_reminder,
        lazy_prompt=lazy_prompt,
        platform=platform_text, # Included within shell_cmd_prompt/no_shell_cmd_prompt
        shell_cmd_prompt=shell_cmd_prompt,
        shell_cmd_reminder=shell_cmd_reminder,
        language=language,
    )

    # Add system reminder if present
    if prompts.system_reminder:
        system_content += "\n" + prompts.system_reminder.format(
            fence=fence,
            quad_backtick_reminder=quad_backtick_reminder,
            lazy_prompt=lazy_prompt,
            shell_cmd_prompt=shell_cmd_prompt, # Pass again if needed by reminder
            shell_cmd_reminder=shell_cmd_reminder,
            platform=platform_text # Pass again if needed by reminder
        )

    return system_content


# --- Message Formatting Helpers ---

def get_repo_map_messages(repo_content: str, prompts: RepoPrompts) -> List[Dict]:
    """Generate repo map messages using RepoPrompts."""
    if not repo_content:
        return []

    # Use the prefix directly from the prompts object
    repo_prefix = prompts.repo_content_prefix.format(other="other " if prompts.chat_files else "") # Mimic BaseCoder logic
    return [
        dict(role="user", content=repo_prefix + repo_content),
        dict(role="assistant", content="Ok, I won't try and edit those files without asking first.")
    ]


def get_readonly_files_messages(content: str, prompts: RepoPrompts) -> List[Dict]:
    """Generate read-only files messages using RepoPrompts."""
    if not content:
        return []

    # Use the prefix directly from the prompts object
    return [
        dict(role="user", content=prompts.read_only_files_prefix + content),
        dict(role="assistant", content="Ok, I will use these files as references.")
    ]


def get_chat_files_messages(content: str, prompts: RepoPrompts, has_repo_map: bool = False) -> List[Dict]:
    """Generate chat files messages using RepoPrompts."""
    if not content:
        if has_repo_map and prompts.files_no_full_files_with_repo_map:
            return [
                dict(role="user", content=prompts.files_no_full_files_with_repo_map),
                dict(role="assistant", content=prompts.files_no_full_files_with_repo_map_reply)
            ]
        return [
            dict(role="user", content=prompts.files_no_full_files),
            dict(role="assistant", content="Ok.") # Use the standard reply
        ]

    # Use prefixes/replies directly from the prompts object
    return [
        dict(role="user", content=prompts.files_content_prefix + content),
        dict(role="assistant", content=prompts.files_content_assistant_reply)
    ]


# --- File Content Formatting ---
def format_files_content(files: List[str], directory: str, fence: Tuple[str, str]) -> str:
    """Format file contents with their relative paths."""
    if not files:
        return ""

    content = []
    for fname in files:
        # Use the provided fence for consistency
        abs_path = os.path.abspath(os.path.join(directory, fname))
        try:
            # Basic check for image files, skip content for them
            if Path(fname).suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                 content.append(f"{fname}\n{fence[0]}\n[Image file content not shown]\n{fence[1]}")
                 continue

            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            # Ensure content ends with a newline before the fence
            if file_content and not file_content.endswith('\n'):
                file_content += '\n'
            content.append(f"{fname}\n{fence[0]}\n{file_content}{fence[1]}")
        except Exception as e:
            print(f"Warning: Could not read or format {fname}: {e}", file=sys.stderr)
            content.append(f"{fname}\n{fence[0]}\n[Error reading file content]\n{fence[1]}")

    # Join with newlines, ensuring a trailing newline if content exists
    result = "\n".join(content)
    return result + "\n" if result else ""


# --- PromptBuilder Class ---

class PromptBuilder:
    """Builds prompts using Aider's formulation."""

    def __init__(
        self,
        root_dir: str,
        map_tokens: int = 4096,
        tokenizer: str = "cl100k_base",
        chat_files: Optional[List[str]] = None,
        read_only_files: Optional[List[str]] = None,
        user_message: str = "",
        extra_mentioned_files: Optional[List[str]] = None,
        extra_mentioned_idents: Optional[List[str]] = None,
        verbose: bool = False,
        no_shell: bool = False,
        fence_open: str = "```",
        fence_close: str = "```",
        history_file: Optional[str] = None, # Add history file path
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.map_tokens = map_tokens
        self.tokenizer = tokenizer
        self.chat_files = chat_files or []
        self.read_only_files = read_only_files or []
        self.user_message = user_message
        self.extra_mentioned_files = extra_mentioned_files or []
        self.extra_mentioned_idents = extra_mentioned_idents or []
        self.verbose = verbose
        self.no_shell = no_shell
        self.fence = (fence_open, fence_close)
        self.history_file = history_file # Store history file path

        # Initialize RepoPrompts from the external file
        self.prompts = RepoPrompts()
        # Attach chat_files to prompts object like BaseCoder does for repo_content_prefix formatting
        self.prompts.chat_files = self.chat_files

        # Initialize RepoMapper
        self.mapper = RepoMapper(
            root_dir=self.root_dir,
            map_tokens=self.map_tokens,
            tokenizer=self.tokenizer,
            verbose=self.verbose
        )

    def build_prompt_messages(self) -> List[Dict]:
        """Constructs the final list of messages for the LLM."""
        chunks = ChatChunks()

        # --- System Prompt ---
        platform_text = get_platform_info(repo=True) # Assume repo context
        system_content = fmt_system_prompt(
            self.prompts,
            fence=self.fence,
            platform_text=platform_text,
            suggest_shell_commands=not self.no_shell
        )
        chunks.system = [{"role": "system", "content": system_content}]

        # --- Few-Shot Examples ---
        chunks.examples = []
        for msg in self.prompts.example_messages:
            example_content = msg["content"].format(fence=self.fence)
            chunks.examples.append({
                "role": msg["role"],
                "content": example_content
            })

        # --- Load History (Compromise: Read from file) ---
        # NOTE: Reads the entire history file, no summarization. May exceed token limits.
        chunks.history = parse_history_markdown(self.history_file)
        if self.verbose and chunks.history:
            print(f"Loaded {len(chunks.history)} messages from history file: {self.history_file}", file=sys.stderr)
        elif self.verbose:
            print(f"No history loaded from: {self.history_file}", file=sys.stderr)


        # --- Context Building (Repo Map, Files) ---
        # This context is added *after* the history messages
        chunks.context = []

        # Extract mentions from the user message
        mentioned_files_from_msg = get_file_mentions(
            self.user_message, self.root_dir, self.chat_files, self.read_only_files
        )
        mentioned_idents_from_msg = get_ident_mentions(self.user_message)

        # Combine explicit args with extracted mentions
        all_mentioned_files = sorted(list(set(self.extra_mentioned_files + mentioned_files_from_msg)))
        all_mentioned_idents = sorted(list(set(self.extra_mentioned_idents + mentioned_idents_from_msg)))

        if self.verbose:
            print(f"Mentioned files for map: {all_mentioned_files}")
            print(f"Mentioned idents for map: {all_mentioned_idents}")

        # Generate Repo Map
        repo_map_content = self.mapper.generate_map(
            chat_files=self.chat_files,
            mentioned_files=all_mentioned_files,
            mentioned_idents=all_mentioned_idents
        )
        if repo_map_content:
            chunks.context.extend(get_repo_map_messages(repo_map_content, self.prompts))

        # Add Read-Only Files
        read_only_content = format_files_content(self.read_only_files, self.root_dir, self.fence)
        if read_only_content:
            chunks.context.extend(get_readonly_files_messages(read_only_content, self.prompts))

        # Add Chat Files
        chat_files_content = format_files_content(self.chat_files, self.root_dir, self.fence)
        chunks.context.extend(get_chat_files_messages(
            chat_files_content,
            self.prompts,
            has_repo_map=bool(repo_map_content)
        ))

        # --- Current User Message ---
        chunks.current = [{"role": "user", "content": self.user_message}]

        # Add system reminder if needed (placed after user message in Aider)
        if self.prompts.system_reminder:
            # Re-fetch platform info if needed by reminder template
            platform_text_for_reminder = get_platform_info(repo=True)
            shell_cmd_prompt_text = self.prompts.shell_cmd_prompt.format(platform=platform_text_for_reminder)
            shell_cmd_reminder_text = self.prompts.shell_cmd_reminder.format(platform=platform_text_for_reminder)
            no_shell_cmd_prompt_text = self.prompts.no_shell_cmd_prompt.format(platform=platform_text_for_reminder)
            no_shell_cmd_reminder_text = self.prompts.no_shell_cmd_reminder.format(platform=platform_text_for_reminder)

            current_shell_prompt = shell_cmd_prompt_text if not self.no_shell else no_shell_cmd_prompt_text
            current_shell_reminder = shell_cmd_reminder_text if not self.no_shell else no_shell_cmd_reminder_text

            chunks.current.append({
                "role": "system",
                "content": self.prompts.system_reminder.format(
                    fence=self.fence,
                    quad_backtick_reminder="", # Assuming no quad backticks for now
                    lazy_prompt=self.prompts.lazy_prompt,
                    shell_cmd_prompt=current_shell_prompt,
                    shell_cmd_reminder=current_shell_reminder,
                    platform=platform_text_for_reminder # Pass platform text if needed
                )
            })

        # Combine all chunks
        return chunks.all_messages()


# --- CLI Execution Logic ---

def main():
    """Handles command-line argument parsing and execution."""
    parser = argparse.ArgumentParser(
        description="Build prompts mimicking Aider's formulation. Outputs JSON message list."
    )
    parser.add_argument("--dir", required=True, help="Root directory of the project")
    # map-script is no longer needed as RepoMapper is imported
    parser.add_argument("--map-tokens", type=int, default=4096, help="Max tokens for repo map")
    parser.add_argument("--tokenizer", default="cl100k_base", help="Tokenizer name for repo map")
    parser.add_argument("--chat-files", nargs='*', default=[], help="Relative paths of files in the chat context")
    parser.add_argument("--read-only-files", nargs='*', default=[], help="Relative paths of read-only files")
    # Allow overriding mentioned files/idents, but also extract from message
    parser.add_argument("--extra-mentioned-files", nargs='*', default=[], help="Manually specify additional mentioned files")
    parser.add_argument("--extra-mentioned-idents", nargs='*', default=[], help="Manually specify additional mentioned identifiers")
    parser.add_argument("--user-message", required=True, help="The current user message/request")
    parser.add_argument("--output", help="Optional file path to write the final prompt JSON to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during script execution")
    parser.add_argument("--no-shell", action="store_true", help="Instruct the prompt to disable shell command suggestions")
    # Add fence argument if needed, otherwise default
    parser.add_argument("--fence-open", default="```", help="Opening fence for code blocks")
    parser.add_argument("--fence-close", default="```", help="Closing fence for code blocks")
    # Add argument for history file path
    parser.add_argument("--history-file", default=".aidermacs_history.md", help="Path to the chat history markdown file.")


    args = parser.parse_args()

    # Instantiate the builder with CLI arguments
    builder = PromptBuilder(
        root_dir=args.dir,
        map_tokens=args.map_tokens,
        tokenizer=args.tokenizer,
        chat_files=args.chat_files,
        read_only_files=args.read_only_files,
        user_message=args.user_message,
        extra_mentioned_files=args.extra_mentioned_files,
        extra_mentioned_idents=args.extra_mentioned_idents,
        verbose=args.verbose,
        no_shell=args.no_shell,
        fence_open=args.fence_open,
        fence_close=args.fence_close,
        history_file=args.history_file, # Pass history file path
    )

    # Generate the messages
    final_prompt_messages = builder.build_prompt_messages()

    # Output the JSON result
    output_json = json.dumps(final_prompt_messages, indent=2)


    if args.output:
        try:
            with open(args.output, "w", encoding='utf-8') as f:
                f.write(output_json)
            print(f"Prompt written to {args.output}")
        except IOError as e:
            print(f"Error writing to output file {args.output}: {e}", file=sys.stderr)
    else:
        # Print to stdout if no output file specified
        print(output_json)


if __name__ == "__main__":
    main()

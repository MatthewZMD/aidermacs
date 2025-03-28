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
import os
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
from repomapper import RepoMapper


# --- Helper Functions (adapted from BaseCoder) ---

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


def get_addable_relative_files(root_dir, chat_files, read_only_files):
    """Simulates BaseCoder's get_addable_relative_files."""
    all_files = set(get_all_relative_files(root_dir, chat_files, read_only_files))
    inchat_files = set(chat_files)
    readonly_files = set(read_only_files)
    return sorted(list(all_files - inchat_files - readonly_files))


def get_file_mentions(content, root_dir, chat_files, read_only_files):
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


def get_ident_mentions(text):
    """Extracts potential identifiers (words) from text, adapted from BaseCoder."""
    # Split on non-alphanumeric characters
    words = set(re.split(r"\W+", text))
    # Filter out short words or purely numeric strings
    return [word for word in words if len(word) >= 3 and not word.isdigit()]


# --- ChatChunks Class (Structure) ---

class ChatChunks:
    """Structures the prompt with clear separation between examples and conversation"""
    def __init__(self):
        self.system = []          # System instructions
        self.examples = []        # Few-shot examples
        self.context = []         # Conversation context (repo map, files)
        self.current = []         # Current user message and response

    def all_messages(self):
        """Combine all message chunks with clear separation"""
        messages = []
        messages.extend(self.system)

        # Add few-shot examples with clear separator
        if self.examples:
            messages.extend(self.examples)
            messages.append({
                "role": "user",
                "content": "Now working with a new code base. The examples above were just demonstrations."
            })
            messages.append({
                "role": "assistant",
                "content": "Understood, I'll focus on the actual files and requests."
            })

        messages.extend(self.context)
        messages.extend(self.current)
        return messages

def get_platform_info(repo=None):
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


def fmt_system_prompt(prompts, fence, platform_text, suggest_shell_commands=True):
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


# --- Message Formatting Functions ---

def get_repo_map_messages(repo_content, prompts):
    """Generate repo map messages using RepoPrompts."""
    if not repo_content:
        return []

    # Use the prefix directly from the prompts object
    repo_prefix = prompts.repo_content_prefix.format(other="other " if prompts.chat_files else "") # Mimic BaseCoder logic
    return [
        dict(role="user", content=repo_prefix + repo_content),
        dict(role="assistant", content="Ok, I won't try and edit those files without asking first.")
    ]

def get_readonly_files_messages(content, prompts):
    """Generate read-only files messages using RepoPrompts."""
    if not content:
        return []

    # Use the prefix directly from the prompts object
    return [
        dict(role="user", content=prompts.read_only_files_prefix + content),
        dict(role="assistant", content="Ok, I will use these files as references.")
    ]

def get_chat_files_messages(content, prompts, has_repo_map=False):
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


# --- Script Execution ---



def format_files_content(files, directory, fence):
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


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Build prompts mimicking Aider's formulation."
    )
    parser.add_argument("--dir", required=True, help="Root directory of the project")
    parser.add_argument("--map-script", default="repomap.py", help="Path to the repomap.py script")
    parser.add_argument("--map-tokens", type=int, default=4096, help="Max tokens for repo map")
    parser.add_argument("--tokenizer", default="cl100k_base", help="Tokenizer name for repomap.py")
    parser.add_argument("--chat-files", nargs='*', default=[], help="Relative paths of files in the chat")
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


    args = parser.parse_args()
    repo_root = os.path.abspath(args.dir)
    fence = (args.fence_open, args.fence_close)

    # Initialize RepoPrompts from the external file
    prompts = RepoPrompts()
    # Attach chat_files to prompts object like BaseCoder does for repo_content_prefix formatting
    prompts.chat_files = args.chat_files

    # Initialize ChatChunks
    chunks = ChatChunks()

    # --- System Prompt ---
    platform_text = get_platform_info(repo=True) # Assume repo context for platform info
    system_content = fmt_system_prompt(
        prompts,
        fence=fence,
        platform_text=platform_text,
        suggest_shell_commands=not args.no_shell
    )
    # BaseCoder structure: Use system role if model supports it (assume yes for standalone)
    chunks.system = [{"role": "system", "content": system_content}]

    # --- Few-Shot Examples ---
    # Format the example conversations using the current fence style
    chunks.examples = []
    for msg in prompts.example_messages:
        example_content = msg["content"].format(fence=fence)
        chunks.examples.append({
            "role": msg["role"],
            "content": example_content
        })


    # --- Repo Map ---
    # Extract mentions from the user message
    mentioned_files_from_msg = get_file_mentions(args.user_message, repo_root, args.chat_files, args.read_only_files)
    mentioned_idents_from_msg = get_ident_mentions(args.user_message)

    # Combine explicit args with extracted mentions
    all_mentioned_files = sorted(list(set(args.extra_mentioned_files + mentioned_files_from_msg)))
    all_mentioned_idents = sorted(list(set(args.extra_mentioned_idents + mentioned_idents_from_msg)))

    if args.verbose:
        print(f"Mentioned files for map: {all_mentioned_files}")
        print(f"Mentioned idents for map: {all_mentioned_idents}")

    mapper = RepoMapper(
        root_dir=repo_root,
        map_tokens=args.map_tokens,
        tokenizer=args.tokenizer,
        verbose=args.verbose
    )
    repo_map_content = mapper.generate_map(
        chat_files=args.chat_files,
        mentioned_files=all_mentioned_files,
        mentioned_idents=all_mentioned_idents
    )
    chunks.repo = get_repo_map_messages(repo_map_content, prompts)

    # --- Read-Only Files ---
    read_only_content = format_files_content(args.read_only_files, repo_root, fence)
    chunks.readonly_files = get_readonly_files_messages(read_only_content, prompts)

    # --- Chat Files ---
    chat_files_content = format_files_content(args.chat_files, repo_root, fence)
    chunks.chat_files = get_chat_files_messages(
        chat_files_content,
        prompts,
        has_repo_map=bool(repo_map_content)
    )

    # --- Current Conversation ---
    chunks.context = []

    # Add repo map messages to context
    repo_map_content = mapper.generate_map(
        chat_files=args.chat_files,
        mentioned_files=all_mentioned_files,
        mentioned_idents=all_mentioned_idents
    )
    if repo_map_content:
        chunks.context.extend(get_repo_map_messages(repo_map_content, prompts))

    # Add read-only files to context
    read_only_content = format_files_content(args.read_only_files, repo_root, fence)
    if read_only_content:
        chunks.context.extend(get_readonly_files_messages(read_only_content, prompts))

    # Add chat files to context
    chat_files_content = format_files_content(args.chat_files, repo_root, fence)
    chunks.context.extend(get_chat_files_messages(
        chat_files_content,
        prompts,
        has_repo_map=bool(repo_map_content)
    ))

    # Add current user message
    chunks.current = [{"role": "user", "content": args.user_message}]

    # Add system reminder if needed
    if prompts.system_reminder:
        chunks.current.append({
            "role": "system",
            "content": prompts.system_reminder.format(
                fence=fence,
                quad_backtick_reminder="",
                lazy_prompt=prompts.lazy_prompt,
                shell_cmd_prompt=prompts.shell_cmd_prompt.format(platform=platform_text),
                shell_cmd_reminder=prompts.shell_cmd_reminder.format(platform=platform_text),
                platform=platform_text
            )
        })


    # --- Output ---
    final_prompt_messages = chunks.all_messages()
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

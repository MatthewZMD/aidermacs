<p align="center">
  <img style='height: auto; width: 40%; object-fit: contain' src="./aidermacs.png">
</p>

# Aidermacs: AI Pair Programming in Emacs

[![MELPA](https://melpa.org/packages/aidermacs-badge.svg)](https://melpa.org/#/aidermacs)
[![MELPA Stable](https://stable.melpa.org/packages/aidermacs-badge.svg)](https://stable.melpa.org/#/aidermacs)
[![NonGNU-devel ELPA](https://elpa.nongnu.org/nongnu-devel/aidermacs.svg)](https://elpa.nongnu.org/nongnu-devel/aidermacs.html)
[![NonGNU ELPA](https://elpa.nongnu.org/nongnu/aidermacs.svg)](https://elpa.nongnu.org/nongnu/aidermacs.html)
[![EMACS](https://img.shields.io/badge/Emacs-26.1-922793?logo=gnu-emacs&logoColor=b39ddb&.svg)](https://www.gnu.org/savannah-checkouts/gnu/emacs/emacs.html)
[![LICENSE](https://img.shields.io/github/license/MatthewZMD/aidermacs?logo=apache&.svg)](https://github.com/MatthewZMD/aidermacs/blob/main/LICENSE)
[![CONTRIBUTORS](https://img.shields.io/github/contributors/MatthewZMD/aidermacs.svg)](https://github.com/MatthewZMD/aidermacs/graphs/contributors)

Aidermacs brings AI-powered development to Emacs by integrating [Aider](https://github.com/paul-gauthier/aider), one of the most powerful open-source AI pair programming tools. If you're missing [Cursor](https://cursor.sh) but prefer living in Emacs, Aidermacs provides similar AI capabilities while staying true to Emacs workflows.

<img style='height: auto; width: 80%; object-fit: contain' src="./introscreen.png">

### Key Features

- Intelligent model selection with multiple backends
- Built-in Ediff integration for AI-generated changes
- Enhanced file management from Emacs
- Great customizability and flexible ways to add content

### Community Speaks

Here's what the community is saying about Aidermacs:

> "Are you using aidermacs? For me superior to cursor." - u/berenddeboer

> "This is amazing... every time I upgrade my packages I see your new commits. I feel this the authentic aider for emacs" - u/wchmbo

> "Between Aidermacs and Gptel it's wild how bleeding edge Emacs is with this stuff. My workplace is exploring MCP registries and even clients that are all the rage (E.g Cursor) lag behind what I can do with mcp.el and gptel for tool use." - u/no_good_names_avail

> "This looks amazing... I have been using ellama with local llms, looks like that will work here too. Great stuff!!" - u/lugpocalypse

> "Honestly huge fan of this. Thank you for the updates!" - u/ieoa

## Quick Start

1. Requirements
- Emacs ≥ 26.1
- [Aider](https://aider.chat/docs/install.html)
- [Transient](https://github.com/magit/transient)
2. Download Aidermacs through [Melpa](https://melpa.org/#/aidermacs) or [Non-GNU Elpa](https://elpa.nongnu.org/nongnu/aidermacs.html), or clone manually
3. Modify this **sample config** and place it in your Emacs `init.el`:
```emacs-lisp
(use-package aidermacs
  :bind (("C-c a" . aidermacs-transient-menu))
  :config
  ; Set API_KEY in .bashrc, that will automatically picked up by aider or in elisp
  (setenv "ANTHROPIC_API_KEY" "sk-...")
  ; defun my-get-openrouter-api-key yourself elsewhere for security reasons
  (setenv "OPENROUTER_API_KEY" (my-get-openrouter-api-key))
  :custom
  ; See the Configuration section below
  (aidermacs-default-chat-mode 'architect)
  (aidermacs-default-model "sonnet"))
```
4. Open a project and run `M-x aidermacs-transient-menu` or `SPC a a` (or your chosen binding).
5. Add files and start coding with AI!

### Spacemacs

For **Spacemacs** users:

1. Add `aidermacs` to your `dotspacemacs-additional-packages` list in your `.spacemacs` file:
```emacs-lisp
dotspacemacs-additional-packages '(
  (aidermacs :variables
              aidermacs-default-chat-mode 'architect
              aidermacs-default-model "sonnet")
)
```

2. Add the keybinding to your `dotspacemacs/user-config` function in `.spacemacs`:
```emacs-lisp
(defun dotspacemacs/user-config ()
  ;; Set leader key for Aidermacs
  (spacemacs/set-leader-keys "aa" 'aidermacs-transient-menu) ; Example binding SPC a a
)
```
3. Open a project and run `M-x aidermacs-transient-menu` or `SPC a a` (or your chosen binding).
4. Add files and start coding with AI!

## Usage

### Getting Started

The main interface to Aidermacs is through its transient menu system (similar to Magit). Access it with:

```
M-x aidermacs-transient-menu
```

Or bind it to a key in your config:

```emacs-lisp
(global-set-key (kbd "C-c a") 'aidermacs-transient-menu)
```

Once the transient menu is open, you can navigate and execute commands using the displayed keys. Here's a summary of the main menu structure:

##### Core
- `a`: Start/Open Session (auto-detects project root)
- `.`: Start in Current Directory (good for monorepos)
- `l`: Clear Chat History
- `s`: Reset Session
- `x`: Exit Session

##### Persistent Modes
- `1`: Code Mode
- `2`: Chat/Ask Mode
- `3`: Architect Mode
- `4`: Help Mode

##### Utilities
- `^`: Show Last Commit (if auto-commits enabled)
- `u`: Undo Last Commit (if auto-commits enabled)
- `R`: Refresh Repo Map
- `h`: Session History
- `o`: Change Main Model
- `v`: Send Voice Command
- `W`: Fetch Web Content
- `?`: Aider Meta-level Help

##### File Actions
- `f`: Add File (C-u: read-only)
- `F`: Add Current File
- `d`: Add From Directory (same type)
- `w`: Add From Window
- `m`: Add From Dired (marked)
- `j`: Drop File
- `J`: Drop Current File
- `k`: Drop From Dired (marked)
- `K`: Drop All Files
- `S`: Create Session Scratchpad
- `G`: Add File to Session
- `A`: List Added Files

##### Code Actions
- `c`: Code Change
- `e`: Question Code
- `r`: Architect Change
- `q`: General Question
- `p`: Question This Symbol
- `g`: Accept Proposed Changes
- `i`: Implement TODO
- `t`: Write Test
- `T`: Fix Test
- `!`: Debug Exception

The `All File Actions` and `All Code Actions` entries open submenus with more specialized commands. Use the displayed keys to navigate these submenus.

### File Management and AI Interaction

When using Aidermacs, you have the flexibility to decide which files the AI should read and edit. Here are some guidelines:

- **Editable Files**: Add files you want the AI to potentially edit. This grants the AI permission to both read and modify these files if necessary.
- **Read-Only Files**: If you want the AI to read a file without editing it, you can add it as read-only. In Aidermacs, all add file commands can be prefixed with `C-u` to specify read-only access.
- **Session Scratchpads**: Use the session scratchpads (`S`) to paste notes or documentation that will be fed to the AI as read-only.
- **External Files**: The "Add file to session" (`G`) command allows you to include files outside the current project (or files in `.gitignore`), as Aider doesn't automatically include these files in its context.

The AI can sometimes determine relevant files on its own, depending on the model and the context of the codebase. However, for precise control, it's often beneficial to manually specify files, especially when dealing with complex projects.

Aider encourages a collaborative approach, similar to working with a human co-worker. Sometimes the AI will need explicit guidance, while other times it can infer the necessary context on its own.

### Prompt Files Minor Mode

Aidermacs provides a minor mode that makes it easy to work with prompt files and other Aider-related files.
When enabled, the minor mode provides these convenient keybindings:

- `C-c C-n` or `C-<return>`: Send line/region line-by-line
- `C-c C-c`: Send block/region as whole
- `C-c C-z`: Switch to Aidermacs buffer

The minor mode is automatically enabled for:
- `.aider.prompt.org` files (create with `M-x aidermacs-open-prompt-file`)
- `.aider.chat.md` files
- `.aider.chat.history.md` files
- `.aider.input.history` files

## Configuration

#### Pre-Run Hook

You can use the `aidermacs-before-run-backend-hook` to run custom setup code before starting the Aider backend. This is particularly useful for:

- Setting environment variables
- Injecting secrets
- Performing any other pre-run configuration

Example usage to securely set an OpenAI API key from password-store:

```elisp
(add-hook 'aidermacs-before-run-backend-hook
          (lambda ()
            (setenv "OPENAI_API_KEY" (password-store-get "code/openai_api_key"))))
```

The environment variable set in the hook would only be visible to aider process. This approach keeps sensitive information out of your dotfiles while still making it available to Aidermacs.

### Default Model Selection

You can customize the default AI model used by Aidermacs. The model is determined in the following order of priority:

1.  The value of the Emacs variable `aidermacs-default-model` if you have set it (e.g., via `setq`).
2.  The value of the `AIDER_MODEL` environment variable, if set.
3.  A built-in default value (e.g., `"sonnet"`).

To set it in Emacs:
```emacs-lisp
(setq aidermacs-default-model "sonnet")
```

This enables easy switching between different AI models without modifying the `aidermacs-extra-args` variable.

*Note: This configuration will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

### Dynamic Model Selection

Aidermacs offers intelligent model selection for solo (non-Architect) mode, automatically detecting and integrating with multiple AI providers:

- Automatically fetches available models from supported providers (OpenAI, Anthropic, DeepSeek, Google Gemini, OpenRouter)
- Caches model lists for quick access
- Supports both popular pre-configured models and dynamically discovered ones
- Handles API keys and authentication automatically from your .bashrc
- Provides model compatibility checking

The dynamic model selection is only for the solo (non-Architect) mode.

To change models in solo mode:
1. Use `M-x aidermacs-change-model` or press `o` in the transient menu
2. Select from either:
   - Popular pre-configured models (fast)
   - Dynamically fetched models from all supported providers (comprehensive)

The system will automatically filter models to only show ones that are:
- Supported by your current Aider version
- Available through your configured API keys
- Compatible with your current workflow

### Architect Mode - Separating Code Reasoning and Editing Models

Aidermacs features an experimental mode using two specialized models for each coding task: an Architect model for reasoning and an Editor model for code generation. This approach has **achieved state-of-the-art (SOTA) results on aider's code editing benchmark**, as detailed in [this blog post](https://aider.chat/2024/09/26/architect.html).

To start new sessions in Architect mode by default, set `aidermacs-default-chat-mode` to `'architect`.

`aidermacs-use-architect-mode` is **deprecated** and will be removed in a future release.

`architect` mode will default to using `aidermacs-default-model`. You may also configure the `aidermacs-architect-model` variable to specify the model to use for the Architect role if you prefer a different main model when using `architect` mode.

By default, the `aidermacs-editor-model` is the same as `aidermacs-default-model`. You only need to set `aidermacs-editor-model` if you want to use a different model for the Editor role. This variable also respects the `AIDER_EDITOR_MODEL` environment variable, following the same priority as `aidermacs-default-model`.

```emacs-lisp
(setq aidermacs-default-chat-mode 'architect)
```

You can switch to it persistently by `M-x aidermacs-switch-to-architect-mode` (`3` in `aidermacs-transient-menu`), or temporarily with `M-x aidermacs-architect-this-code` (`r` in `aidermacs-transient-menu`).

You can configure each model independently:

```emacs-lisp
;; Default model used for all modes unless overridden
(setq aidermacs-default-model "sonnet")

;; Optional: Set specific model for architect reasoning
(setq aidermacs-architect-model "deepseek/deepseek-reasoner")

;; Optional: Set specific model for code generation
(setq aidermacs-editor-model "deepseek/deepseek-chat")
```

The model hierarchy works as follows:
- When Architect mode is enabled:
    - The Architect model handles high-level reasoning and solution design
    - The Editor model executes the actual code changes
- When Architect mode is disabled, only `aidermacs-default-model` is used
- You can configure specific models or let them automatically use the default model

Models will reflect changes to `aidermacs-default-model` unless they've been explicitly set to a different value.

*Note: These configurations will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

### Customize Weak Model

The Weak model is used for commit messages (if you have `aidermacs-auto-commits` set to `t`) and chat history summarization (default depends on –model). You can customize it using the `aidermacs-weak-model` variable, which also respects the `AIDER_WEAK_MODEL` environment variable.

```emacs-lisp
;; default to nil
(setq aidermacs-weak-model "deepseek/deepseek-chat")
```

You can change the Weak model during a session by using `C-u o` (`aidermacs-change-model` with a prefix argument). In most cases, you won't need to change this as Aider will automatically select an appropriate Weak model based on your main model.

*Note: These configurations will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

#### Architect Mode Confirmation

By default, Aidermacs requires explicit confirmation before applying changes proposed in Architect mode. This gives you a chance to review the AI's plan before any code is modified.

If you prefer to automatically accept all Architect mode changes without confirmation (similar to Aider's default behavior), you can enable this with:

```emacs-lisp
(setq aidermacs-auto-accept-architect t)
```

*Note: These configurations will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

### Terminal Backend Selection

Choose your preferred terminal backend by setting `aidermacs-backend`:

`vterm` offers better terminal compatibility, while `comint` provides a simple, built-in option that remains fully compatible with Aidermacs.

```emacs-lisp
;; Use vterm backend (default is comint)
(setq aidermacs-backend 'vterm)
```

Available backends:
- `comint` (default): Uses Emacs' built-in terminal emulation
- `vterm`: Leverages vterm for better terminal compatibility

### Emacs theme support
The vterm backend will use the faces defined by your active Emacs theme to set the colors for aider. It tries to guess some reasonable color values based on your themes. In some cases this will not work perfectly; if text is unreadable for you, you can turn this off as follows:

``` emacs-lisp
;; don't match emacs theme colors
(setopt aidermacs-vterm-use-theme-colors nil)
```

### Multiline Input Configuration

You can customize keybindings for multiline input, this key allows you to enter multiple lines without sending the command to Aider. Press `RET` normally to send the command.

```emacs-lisp
;; Comint backend:
(setq aidermacs-comint-multiline-newline-key "S-<return>")
;; Vterm backend:
(setq aidermacs-vterm-multiline-newline-key "S-<return>")
```

### Remote File Support with Tramp

Aidermacs fully supports working with remote files through Emacs' Tramp mode. This allows you to use Aidermacs on files hosted on remote servers via SSH, Docker, and other protocols supported by Tramp.

When working with remote files:
- File paths are automatically localized for the remote system
- All Aidermacs features work seamlessly across local and remote files
- Edits are applied directly to the remote files
- Diffs and change reviews work as expected

Example usage:
```emacs-lisp
;; Open a remote file via SSH
(find-file "/ssh:user@remotehost:/path/to/file.py")

;; Start Aidermacs session - it will automatically detect the remote context
M-x aidermacs-transient-menu
```

### Prompt Selection and History

Aidermacs makes it easy to reuse prompts through:

1. **Prompt History** - Your previously used prompts are saved and can be quickly selected
2. **Common Prompts** - A curated list of frequently used prompts for common tasks defined in `aidermacs-common-prompts`:

When entering a prompt, you can:
- Select from your history or common prompts using completion
- Still type custom prompts when needed

The prompt history and common prompts are available across all sessions.

### File Watching

Enable watching for AI coding instructions in your repository files with `aidermacs-watch-files`:

```emacs-lisp
;; Enable file watching
(setq aidermacs-watch-files t)
```

When enabled, Aidermacs will will watch all files in your repo and look for any AI coding instructions you add using your favorite IDE or text editor.

Specifically, aider looks for one-liner comments (`# ...`, `// ...` or `-- ...`, regardless of the comment style that language supports) that either start or end with `AI`, `AI!` or `AI?` like these:
```
# Make a snake game. AI!
# What is the purpose of this method AI?

// Write a protein folding prediction engine. AI!
```
Aidermacs will take note of all the comments that start or end with `AI`. Comments that include `AI!` with an exclamation point or `AI?` with a question mark are special. They trigger aider to take action to collect all the AI comments and use them as your instructions.

- `AI!` triggers aider to make changes to your code.
- `AI?` triggers aider to answer your question.

*Note: This feature currently only works in the vterm mode.*

*Note: These configurations will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

### Read-Only Files

You can configure certain files to always be added as read-only when starting any Aidermacs session by setting the `aidermacs-global-read-only-files` and `aidermacs-project-read-only-files`.

```emacs-lisp
;; Always add these files as read-only to all Aidermacs sessions
;; For files that exists outside the project directory
(setq aidermacs-global-read-only-files '("~/.aider/AI_RULES.md"))
;; For files that exists within the project directory
(setq aidermacs-project-read-only-files '("CONVENTIONS.md" "README.md"))
```

When an Aidermacs session starts, these files will be automatically added as read-only if they exist in the project. This is useful for documentation files or other references that you want the AI to be aware of but not modify.

### Diff and Change Review

Control whether to show diffs for AI-generated changes with `aidermacs-show-diff-after-change`:

```emacs-lisp
;; Enable/disable showing diffs after changes (default: t)
(setq aidermacs-show-diff-after-change t)
```

When enabled, Aidermacs will:
- Capture the state of files before AI edits
- Show diffs using Emacs' built-in ediff interface
- Allow you to review and accept/reject changes

### Re-Enable Auto-Commits

Aider automatically commits AI-generated changes by default. We consider this behavior *very* intrusive, so we've disabled it. You can re-enable auto-commits by setting `aidermacs-auto-commits` to `t`:

```emacs-lisp
;; Enable auto-commits
(setq aidermacs-auto-commits t)
```

With auto-commits disabled, you must manually commit changes using your preferred Git workflow.

*Note: This configuration will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

### Control Buffer Killing on Exit

By default, when you exit an Aidermacs session using `aidermacs-exit` (or `x` in the transient menu), the Aidermacs buffer remains open. If you prefer the buffer to be automatically killed upon exiting the session, you can customize this behavior:

```emacs-lisp
;; Kill the Aider buffer when exiting the session
(setq aidermacs-exit-kills-buffer t)
```

### Customize Aider Options with `aidermacs-extra-args`

If these configurations aren't sufficient, the `aidermacs-extra-args` variable enables passing any Aider-supported command-line options.

See the [Aider configuration documentation](https://aider.chat/docs/config/options.html) for a full list of available options.

```emacs-lisp
;; Set the verbosity:
(add-to-list 'aidermacs-extra-args "--verbose")
```

These arguments will be appended to the Aider command when it is run. Note that the `--model` argument is automatically handled by `aidermacs-default-model` and should not be included in `aidermacs-extra-args`.

### Overwrite Configuration with Configuration File

Aidermacs supports project-specific configurations via `.aider.conf.yml` files. To enable this:

1.  Create a `.aider.conf.yml` in your home dir, project's root, or the current directory, defining your desired settings. See the [Aider documentation](https://aider.chat/docs/config/aider_conf.html) for available options.

2.  Tell Aidermacs to use the config file in one of two ways:

    ```emacs-lisp
    ;; Set the `aidermacs-config-file` variable in your Emacs config:
    (setq aidermacs-config-file "/path/to/your/project/.aider.conf.yml")
    ;; *Or*, include the `--config` or `-c` flag in `aidermacs-extra-args`:
    (setq aidermacs-extra-args '("--config" "/path/to/your/project/.aider.conf.yml"))
    ```

*Note: You can also rely on Aider's default behavior of automatically searching for `.aider.conf.yml` in the home directory, project root, or current directory, in that order. In this case, you do not need to set `aidermacs-config-file` or include `--config` in `aidermacs-extra-args`.*

* **Important: Configuration Precedence**

When an Aidermacs session starts, it determines which settings to use based on the following hierarchy:

1.  **Aider Configuration File (`.aider.conf.yml`):** If a config file is found (either in a default location or specified via `aidermacs-config-file` or `aidermacs-extra-args`), it takes **highest priority**. Aidermacs will **ignore** its own Emacs variables (like `aidermacs-default-model`) and let the `aider` tool manage settings directly from the file. The precedence is then determined by `aider` itself (where command-line arguments override environment variables, which override the config file).

2.  **Emacs Configuration (if no config file is used):** If no `.aider.conf.yml` is present, Aidermacs will build the command-line arguments for `aider`. The values it uses are determined by this order of priority:
    a. **Emacs Variables (Highest):** Values you set in your `init.el` (e.g., `(setq aidermacs-default-model "...")`).
    b. **Environment Variables:** If an Emacs variable is not set, its value is taken from the corresponding environment variable (e.g., `AIDER_MODEL`).
    c. **Built-in Defaults (Lowest):** If neither of the above are set, a hardcoded default is used.

**To avoid conflicts:** When using a `.aider.conf.yml` file, you should define all your settings within that file. Do not set variables like `aidermacs-default-model` in Emacs or pass arguments like `--model` in `aidermacs-extra-args`, as they will be ignored and can cause confusion.

### Claude 3.7 Sonnet Thinking Tokens

Aider can work with Sonnet 3.7's [new thinking tokens](https://www.anthropic.com/news/claude-3-7-sonnet). You can now enable and configure thinking tokens more easily using the following methods:

1.  **In-Chat Command:** Use the `/think-tokens` command followed by the desired token budget. For example: `/think-tokens 8k` or `/think-tokens 10000`. Supported formats include `8096`, `8k`, `10.5k`, and `0.5M`.

2.  **Command-Line Argument:** Set the `--thinking-tokens` argument when starting Aidermacs. For example, you can add this to your `aidermacs-extra-args`:

    ```emacs-lisp
    (setq aidermacs-extra-args '("--thinking-tokens" "16k"))
    ```

These methods provide a more streamlined way to control thinking tokens without requiring manual configuration of `.aider.model.settings.yml` files.

*Note: If you are using an `.aider.conf.yml` file, you can also set the `thinking_tokens` option there.*

#### Working with Prompt Files

The `.aider.prompt.org` file is particularly useful for:
- Storing frequently used prompts
- Documenting common workflows
- Quick access to complex instructions

You can customize which files automatically enable the minor mode by configuring `aidermacs-auto-mode-files`:

```emacs-lisp
(setq aidermacs-auto-mode-files
      '(".aider.prompt.org"
        ".aider.chat.md"
        ".aider.chat.history.md"
        ".aider.input.history"
        "my-custom-aider-file.org"))  ; Add your own files
```


## FAQ

### Aider?
Please check [Aider's FAQ](https://aider.chat/docs/faq.html) for Aider related questions.

### Can I use my own AI models?
Yes! Aidermacs supports any OpenAI-compatible API endpoint. Check Aider documentation on [Ollama](https://aider.chat/docs/llms/ollama.html) and [LiteLLM](https://aider.chat/docs/llms/other.html).

### Is my code sent to the AI provider?
Yes, the code you add to the session is sent to the AI provider. Be mindful of sensitive code.

### Why aider not support Python 3.13
Aider only support Python 3.12 currently, you can use [uv](https://github.com/astral-sh/uv) install aider:

```bash
uv tool install --force --python python3.12 aider-chat@latest
```

If you encounter a proxy-related [issue](https://github.com/Aider-AI/aider/issues/2984) , such as the error indicating that the 'socksio' package is not installed, please use:

```bash
uv tool install --force --python python3.12 aider-chat@latest --with 'httpx[socks]'
```

And adjust aidermacs program with below config.

```elisp
(setq aidermacs-program (expand-file-name "~/.local/bin/aider"))
```

### Local models prefer English, even if the prompt language is different

Some local models in aider may default to English, even if you use a different language in your prompt. To ensure the chat language matches your preference (e.g., Russian), add the following to your Emacs configuration:

```emacs-lisp
(add-to-list 'aidermacs-extra-args "--chat-language ru")
```

This will instruct aider to use Russian for chat interactions.

### Aidermacs or Aider ignores command-line arguments and prompts for provider selection when using Ollama

If the `~/.aider` directory exists (even if empty), Aidermacs or Aider may silently ignore command-line arguments and unexpectedly prompt for provider selection (e.g., forcing OpenRouter), even if your configuration is correct for Ollama or other providers.

**Workaround:**
Remove the existing aider directory and restart Emacs:

```bash
rm -rf ~/.aider*
```

After this, Aidermacs should start as expected with your configured provider (e.g., Ollama).

### Aidermacs vs Aider CLI vs aider.el?

Aidermacs is designed to provide a more Emacs-native experience while integrating with [Aider CLI](https://github.com/paul-gauthier/aider). It began as a fork of `aider.el`, but has since diverged significantly to prioritize Emacs workflow integration, built by Emacser, for Emacser.

While aider.el tries to mirror Aider's CLI behavior, Aidermacs is built around Emacs-specific features and paradigms. This design philosophy allows you to harness Aider's powerful capabilities through a natural, Emacs-native coding experience.

With Aidermacs, you get:

1. Built-in Ediff Integration for AI-Generated Changes
   - Seamless Code Review: Automatically shows diffs for all AI-modified files using Emacs' powerful `ediff` interface
   - Familiar Interface: Uses Emacs' native `ediff` workflow for reviewing changes
   - Interactive Workflow: Accept or reject changes with standard `ediff` commands
   - Syntax Highlighting: Maintains proper syntax highlighting during comparisons
   - Safe Change Management: Preserves original file states for easy comparison and rollback

2. Intelligent Model Selection
   - Automatic discovery of available models from multiple providers
   - Real-time model compatibility checking
   - Seamless integration with your configured API keys
   - Caching for quick access to frequently used models
   - Support for both popular pre-configured models and dynamically discovered ones

3. Flexible Terminal Backend Support
   - Aidermacs supports multiple terminal backends (comint and vterm) for better compatibility and performance
   - Easy configuration to choose your preferred terminal emulation
   - Extensible architecture for adding new backends

4. Smarter Syntax Highlighting
   - AI-generated code appears with proper syntax highlighting in major languages.
   - Ensures clarity and readability without additional configuration.

5. Better Support for Multiline Input
   - Aider is designed as a CLI program, where multiline input is restricted by terminal limitations.
   - Terminal-based tools require special syntax or manual formatting to handle multiline input, which can be cumbersome and unintuitive.
   - Aidermacs eliminates these restrictions by handling multiline prompts natively within Emacs, allowing you to compose complex AI requests just like any other text input.
   - Whether you're pasting blocks of code or refining AI-generated responses, multiline interactions in Aidermacs feel natural and seamless.

6. Enhanced File Management from Emacs
   - List files currently in chat with `M-x aidermacs-list-added-files`
   - Drop specific files from chat with `M-x aidermacs-drop-file`
   - View output history with `M-x aidermacs-show-output-history`
   - Interactively select files to add with `M-x aidermacs-add-files-interactively`
   - Add content from any file to a specific session with `M-x aidermacs-add-file-to-session`
   - Create a temporary file for adding code snippets or notes to the Aider session with `M-x aidermacs-create-session-scratchpad`
   - Full support for remote files via Tramp (SSH, Docker, etc.)
   - and more

7. Greater Configurability
   - Aidermacs offers more customization options to tailor the experience to your preferences.

8. Streamlined Transient Menu Selection
   - The transient menus have been completely redesigned to encompass functionality and ergonomics, prioritizing user experience.

9. Flexible Ways to Add Content
   - Aidermacs provides multiple ways to add content to the Aider session, including adding files, creating temporary scratchpad files, and more.

10. Community-Driven Development
    - Aidermacs is actively developed and maintained by the community, incorporating user feedback and contributions.
    - We prioritize features and improvements that directly benefit Emacs users, ensuring a tool that evolves with your needs.

... and more to come 🚀

## Video Demo

[<img src="https://img.youtube.com/vi/fB3-ie6zs4Y/0.jpg" width=600>](https://www.youtube.com/watch?v=fB3-ie6zs4Y)

## Community-Driven Development

Aidermacs thrives on community involvement. We believe collaborative development with user and contributor input creates the best software. We encourage you to:

- Contribute Code: Submit pull requests with bug fixes, new features, or improvements to existing functionality.
- Report Issues: Let us know about any bugs, unexpected behavior, or feature requests through GitHub Issues.
- Share Ideas: Participate in discussions and propose new ideas for making Aidermacs even better.
- Improve Documentation: Help us make the documentation clearer, more comprehensive, and easier to use.

Your contributions are essential for making Aidermacs the best AI pair programming tool in Emacs!

<a href = "https://github.com/MatthewZMD/aidermacs/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo=MatthewZMD/aidermacs"/>
</a>

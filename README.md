<p align="center">
  <img style='height: auto; width: 40%; object-fit: contain' src="./aidermacs.png">
</p>

# Aidermacs: AI Pair Programming in Emacs
[![MELPA](https://melpa.org/packages/aidermacs-badge.svg)](https://melpa.org/#/aidermacs)
[![NonGNU ELPA](https://elpa.nongnu.org/nongnu/aidermacs.svg)](https://elpa.nongnu.org/nongnu/aidermacs.html)
[![EMACS](https://img.shields.io/badge/Emacs-26.1-922793?logo=gnu-emacs&logoColor=b39ddb&.svg)](https://www.gnu.org/savannah-checkouts/gnu/emacs/emacs.html)
[![LICENSE](https://img.shields.io/github/license/MatthewZMD/aidermacs?logo=apache&.svg)](https://github.com/MatthewZMD/aidermacs/blob/master/LICENSE)
[![CONTRIBUTORS](https://img.shields.io/github/contributors/MatthewZMD/aidermacs.svg)](https://github.com/MatthewZMD/aidermacs/graphs/contributors)
[![ISSUES](https://img.shields.io/github/issues/MatthewZMD/aidermacs.svg)](https://github.com/MatthewZMD/aidermacs/issues)

Missing [Cursor](https://cursor.sh) but prefer living in Emacs? Aidermacs brings Cursor-like AI-powered development to your Emacs workflow by integrating [Aider](https://github.com/paul-gauthier/aider), one of the most powerful open-source AI pair programming tools. As a community-driven project, Aidermacs prioritizes Emacs users' needs and preferences while providing the same powerful features found in Cursor!

- Built-in **Ediff** integration for reviewing AI-generated changes
- Top performance on the SWE Bench, solving real GitHub issues in major open source projects
- Support for multi-file edits in complex codebases
- Real-time file synchronization for true pair programming
- Broad language support including Python, JavaScript, TypeScript, PHP, HTML, CSS, etc.
- Compatibility with leading AI models like Claude, ChatGPT, DeepSeek, etc.

## Why Aidermacs?

<img style='height: auto; width: 80%; object-fit: contain' src="./introscreen.png">

### Video Demo

[<img src="https://img.youtube.com/vi/fB3-ie6zs4Y/0.jpg" width=600>](https://www.youtube.com/watch?v=fB3-ie6zs4Y)

### Community-Driven Development

Aidermacs thrives on community involvement. We believe collaborative development with user and contributor input creates the best software. We encourage you to:

- Contribute Code: Submit pull requests with bug fixes, new features, or improvements to existing functionality.
- Report Issues: Let us know about any bugs, unexpected behavior, or feature requests through GitHub Issues.
- Share Ideas: Participate in discussions and propose new ideas for making Aidermacs even better.
- Improve Documentation: Help us make the documentation clearer, more comprehensive, and easier to use.

Your contributions are essential for making Aidermacs the best AI pair programming tool in Emacs!

<a href = "https://github.com/MatthewZMD/aidermacs/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo=MatthewZMD/aidermacs"/>
</a>

#### Community Speaks

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
2. Download Aidermacs through [melpa](https://melpa.org/#/aidermacs), or clone manually
2. Modify this **sample config** and place it in your Emacs `init.el`:
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
  (aidermacs-use-architect-mode t)
  (aidermacs-default-model "sonnet"))
```
3. Open a project and run `M-x aidermacs-transient-menu` or `C-c a` (where you bind it)
4. Add files and start coding with AI!

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

This approach keeps sensitive information out of your dotfiles while still making it available to Aidermacs.

### Default Model Selection

You can customize the default AI model used by Aidermacs by setting the `aidermacs-default-model` variable:

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

To enable this mode, set `aidermacs-use-architect-mode` to `t`. You must also configure the `aidermacs-architect-model` variable to specify the model to use for the Architect role.

By default, the `aidermacs-editor-model` is the same as `aidermacs-default-model`. You only need to set `aidermacs-editor-model` if you want to use a different model for the Editor role.

When Architect mode is enabled, the `aidermacs-default-model` setting is ignored, and `aidermacs-architect-model` and `aidermacs-editor-model` are used instead.

```emacs-lisp
(setq aidermacs-use-architect-mode t)
```
You can switch to it persistently by `M-x aidermacs-switch-to-architect-mode` (`3` in `aidermacs-transient-menu`), or temporarily with `M-x aidermacs-architect-this-code` (`r` in `aidermacs-transient-menu`).

You can configure each model independently:
```emacs-lisp
;; Architect model for reasoning (initially defaults to aidermacs-default-model)
(setq aidermacs-architect-model "sonnet")

;; Editor model for code generation (initially defaults to aidermacs-default-model)
(setq aidermacs-editor-model "deepseek/deepseek-chat")
```

The model hierarchy works as follows:
- When Architect mode is enabled:
    - The Architect model handles high-level reasoning and solution design
    - The Editor model executes the actual code changes
- When Architect mode is disabled, only `aidermacs-default-model` is used
- You can configure both models independently or let them default to `aidermacs-default-model`

*Note: The architect and editor models only inherit from `aidermacs-default-model` when first initialized. Changing `aidermacs-default-model` later will not update the other models. For consistent behavior, set each model explicitly.*

*Note: These configurations will be overwritten by the existence of an `.aider.conf.yml` file (see [details](#Overwrite-Configuration-with-Configuration-File)).*

### Customize Weak Model

The Weak model is used for commit messages (if you have `aidermacs-auto-commits` set to `t`) and chat history summarization (default depends on –model). You can customize it using

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

### Customize Aider Options with `aidermacs-extra-args`

If these configurations aren't sufficient, the `aidermacs-extra-args` variable enables passing any Aider-supported command-line options.

See the [Aider configuration documentation](https://aider.chat/docs/config/options.html) for a full list of available options.

```emacs-lisp
;; Set the verbosity:
(setq aidermacs-extra-args '("--verbose"))
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

* **Important:** When using a config file, all other Aidermacs configuration variables supplying an argument option (e.g., `aidermacs-default-model`, `aidermacs-architect-model`, `aidermacs-use-architect-mode`) are **IGNORED**. Aider will *only* use the settings specified in your `.aider.conf.yml` file. Do not attempt to combine these Emacs settings with a config file, as the results will be unpredictable.
* **Precedence:** Settings in `.aider.conf.yml` *always* take precedence when a config file is explicitly specified.
* **Avoid Conflicts:** When using a config file, *do not* include model-related arguments (like `--model`, `--architect`, etc.) in `aidermacs-extra-args`.  Configure *all* settings within your `.aider.conf.yml` file.

### Claude 3.7 Sonnet Thinking Tokens

Aider can work with Sonnet 3.7's [new thinking tokens](https://www.anthropic.com/news/claude-3-7-sonnet). You can now enable and configure thinking tokens more easily using the following methods:

1.  **In-Chat Command:** Use the `/think-tokens` command followed by the desired token budget. For example: `/think-tokens 8k` or `/think-tokens 10000`. Supported formats include `8096`, `8k`, `10.5k`, and `0.5M`.

2.  **Command-Line Argument:** Set the `--thinking-tokens` argument when starting Aidermacs. For example, you can add this to your `aidermacs-extra-args`:

    ```emacs-lisp
    (setq aidermacs-extra-args '("--thinking-tokens" "16k"))
    ```

These methods provide a more streamlined way to control thinking tokens without requiring manual configuration of `.aider.model.settings.yml` files.

*Note: If you are using an `.aider.conf.yml` file, you can also set the `thinking_tokens` option there.*

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

### Prompt Files Minor Mode

Aidermacs provides a minor mode that makes it easy to work with prompt files and other Aider-related files. The minor mode can be enabled automatically for specific files by calling `(aidermacs-setup-minor-mode)` in your config:

```emacs-lisp
(aidermacs-setup-minor-mode)
```

When enabled, the minor mode provides these convenient keybindings:

- `C-c C-n` or `C-<return>`: Send line/region line-by-line
- `C-c C-c`: Send block/region as whole
- `C-c C-z`: Switch to Aidermacs buffer

The minor mode is automatically enabled for:
- `.aider.prompt.org` files (create with `M-x aidermacs-open-prompt-file`)
- `.aider.chat.md` files
- `.aider.chat.history.md` files
- `.aider.input.history` files

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

## Aidermacs vs aider.el

Aidermacs is designed to provide a more Emacs-native experience while still integrating with [Aider](https://github.com/paul-gauthier/aider). It began as a fork of [aider.el](https://github.com/tninja/aider.el), but has since diverged significantly to prioritize Emacs workflow integration.

While `aider.el` strictly mirrors Aider's CLI behavior, `Aidermacs` is built around Emacs-specific features and paradigms. This design philosophy allows you to harness Aider's powerful capabilities through a natural, Emacs-native coding experience.

With `Aidermacs`, you get:

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
   - `Aidermacs` supports multiple terminal backends (comint and vterm) for better compatibility and performance
   - Easy configuration to choose your preferred terminal emulation
   - Extensible architecture for adding new backends

4. Smarter Syntax Highlighting
   - AI-generated code appears with proper syntax highlighting in major languages.
   - Ensures clarity and readability without additional configuration.

5. Better Support for Multiline Input
   - `aider` is primarily designed as a command-line program, where multiline input is restricted by terminal limitations.
   - Terminal-based tools require special syntax or manual formatting to handle multiline input, which can be cumbersome and unintuitive.
   - `Aidermacs` eliminates these restrictions by handling multiline prompts natively within Emacs, allowing you to compose complex AI requests just like any other text input.
   - Whether you're pasting blocks of code or refining AI-generated responses, multiline interactions in `Aidermacs` feel natural and seamless.

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
   - `Aidermacs` offers more customization options to tailor the experience to your preferences.

8. Streamlined Transient Menu Selection
   - The transient menus have been completely redesigned to encompass functionality and ergonomics, prioritizing user experience.

9. Flexible Ways to Add Content
   - `Aidermacs` provides multiple ways to add content to the Aider session, including adding files, creating temporary scratchpad files, and more.

10. Community-Driven Development
    - `Aidermacs` is actively developed and maintained by the community, incorporating user feedback and contributions.
    - We prioritize features and improvements that directly benefit Emacs users, ensuring a tool that evolves with your needs.

... and more to come 🚀

## FAQ

### Aider?
Please check [Aider's FAQ](https://aider.chat/docs/faq.html) for Aider related questions.

### Can I use my own AI models?
Yes! Aidermacs supports any OpenAI-compatible API endpoint. Check Aider documentation on [Ollama](https://aider.chat/docs/llms/ollama.html) and [LiteLLM](https://aider.chat/docs/llms/other.html).

### Is my code sent to the AI provider?
Yes, the code you add to the session is sent to the AI provider. Be mindful of sensitive code.

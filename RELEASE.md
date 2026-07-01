# RELEASE

# Aidermacs 1.8

## What's Changed
* Refactor model pricing to use local litellm data instead of API queries by @OverbearingPearl in https://github.com/MatthewZMD/aidermacs/pull/200

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.7...v1.8


# Aidermacs 1.7

## What's Changed
* Add model ranking by price in selection UI by @OverbearingPearl in https://github.com/MatthewZMD/aidermacs/pull/181
* Improve tramp support and per-workspace cache variables by @skittishdev in https://github.com/MatthewZMD/aidermacs/pull/175
* feat: add --linear-output option for Aider-CE in aidermacs-run by @MatthewZMD in https://github.com/MatthewZMD/aidermacs/pull/184

## New Contributors
* @skittishdev made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/175

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.6...v1.7


# Aidermacs 1.6

## What's Changed
* Fix regexp for vterm-filter-buffer-substring by @meliache in https://github.com/MatthewZMD/aidermacs/pull/156
* feat: Add aidermacs-run-in-directory with ":" key binding by @veracioux in https://github.com/MatthewZMD/aidermacs/pull/158
* Fix: Use process-file to get aider version. Fixes #155 by @bdf369 in https://github.com/MatthewZMD/aidermacs/pull/161
* UX improve: show current running status in transient menu by @blahgeek in https://github.com/MatthewZMD/aidermacs/pull/163
* Respect `aidermacs-subtree-only` even when `aidermacs-config-file` is set by @blahgeek in https://github.com/MatthewZMD/aidermacs/pull/165
* Make a copy of process-environment to prevent leaking secrets set in hook by @blahgeek in https://github.com/MatthewZMD/aidermacs/pull/166
* Wait for a bit longer time for the backend to be ready by @blahgeek in https://github.com/MatthewZMD/aidermacs/pull/167
* Update README about process-environment by @blahgeek in https://github.com/MatthewZMD/aidermacs/pull/169
* Make code actions automatically select relevant context by @remo5000 in https://github.com/MatthewZMD/aidermacs/pull/176
* Fix Aider version check by @claritystorm in https://github.com/MatthewZMD/aidermacs/pull/177
* Fix aider version check by @Zhen-Ni in https://github.com/MatthewZMD/aidermacs/pull/178
* Update thinking instructions for newer Sonnets by @plpxsk in https://github.com/MatthewZMD/aidermacs/pull/180
* Display model pricing in model selection by @OverbearingPearl in https://github.com/MatthewZMD/aidermacs/pull/181
* Fix: Use real model value in `aidermacs-set-model` by @OverbearingPearl in https://github.com/MatthewZMD/aidermacs/pull/182

## New Contributors
* @meliache made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/156
* @veracioux made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/158
* @bdf369 made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/161
* @blahgeek made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/163
* @Zhen-Ni made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/178
* @plpxsk made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/180
* @OverbearingPearl made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/181

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.5...v1.6


# Aidermacs 1.5

## What's Changed
* backends: Adjust the backends for lazy-load vterm by @brianmcgillion in https://github.com/MatthewZMD/aidermacs/pull/137
* Don't Run block fontification if position has not advanced by @CeleritasCelery in https://github.com/MatthewZMD/aidermacs/pull/140
* feat: aidermacs send voice - fixes #128 by @rajp152k in https://github.com/MatthewZMD/aidermacs/pull/143
* Fix aidermacs--send-command-backend warning during compilation by @AdamNiederer in https://github.com/MatthewZMD/aidermacs/pull/142
* Feat: show list of edited files by @Azkae in https://github.com/MatthewZMD/aidermacs/pull/148
* feat(core)!: introduce `aidermacs-default-chat-mode`; deprecate `aidermacs-use-architect-mode` by @timvisher-dd in https://github.com/MatthewZMD/aidermacs/pull/147
* fix file selection when the previous buffer was not closed by @Azkae in https://github.com/MatthewZMD/aidermacs/pull/150
* feat: support environment variables for default AI model configurations by @timvisher-dd in https://github.com/MatthewZMD/aidermacs/pull/151
* fix default chat mode: defaults to 'code by @Azkae in https://github.com/MatthewZMD/aidermacs/pull/153

## New Contributors
* @brianmcgillion made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/137
* @rajp152k made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/143
* @Azkae made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/148
* @timvisher-dd made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/147

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.4...v1.5


# Aidermacs 1.4

## What's Changed
* [suggestion] Add project file picking by @remo5000 in https://github.com/MatthewZMD/aidermacs/pull/120
* Do not require exact matches when selecting models by @AdamNiederer in https://github.com/MatthewZMD/aidermacs/pull/121
* Don't send commands to aider if it's not ready for them by @AdamNiederer in https://github.com/MatthewZMD/aidermacs/pull/124
* Use read-from-minibuffer instead of completing-read for prompts by @AdamNiederer in https://github.com/MatthewZMD/aidermacs/pull/123
* Don't start aider just to stop it when not running  by @AdamNiederer in https://github.com/MatthewZMD/aidermacs/pull/125
* Use read-string instead of read-from-minibuffer by @AdamNiederer in https://github.com/MatthewZMD/aidermacs/pull/126
* feat: Ability to configure read-only files by @rezaamashi in https://github.com/MatthewZMD/aidermacs/pull/130
* Add verilog support and prevent underscores from changing formatting by @CeleritasCelery in https://github.com/MatthewZMD/aidermacs/pull/131
* Fix cursor moving during syntax highlighting by @CeleritasCelery in https://github.com/MatthewZMD/aidermacs/pull/132
* Fix temp buffer exec path by @toniz4 in https://github.com/MatthewZMD/aidermacs/pull/133

## New Contributors
* @remo5000 made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/120
* @AdamNiederer made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/121
* @rezaamashi made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/130
* @toniz4 made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/133

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.3...v1.4

# Aidermacs 1.3

## What's Changed
* Fix the part about the "AI" comments in the README.md by @Martinsos in https://github.com/MatthewZMD/aidermacs/pull/106
* aidermacs--form-prompt: Always use prompt-prefix and context. by @thanhvg in https://github.com/MatthewZMD/aidermacs/pull/108
* Use `display-warning` when warning user about font-lock position by @benthamite in https://github.com/MatthewZMD/aidermacs/pull/113
* Fix aidermacs typos by @Dima-369 in https://github.com/MatthewZMD/aidermacs/pull/114
* feat: add web content fetching function to aidermacs by @ArthurHeymans in https://github.com/MatthewZMD/aidermacs/pull/115
* feat: add auto-commit command to Magit transient menu by @ArthurHeymans in https://github.com/MatthewZMD/aidermacs/pull/116
* Change syntax highlighting to use a state based approach by @CeleritasCelery in https://github.com/MatthewZMD/aidermacs/pull/118
* Add markdown formatting to the aider session by @CeleritasCelery in https://github.com/MatthewZMD/aidermacs/pull/119

## New Contributors
* @Martinsos made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/106
* @thanhvg made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/108
* @benthamite made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/113
* @Dima-369 made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/114

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.2...v1.3

# Aidermacs 1.2

## What's Changed
* Use uv install aider. by @manateelazycat in https://github.com/MatthewZMD/aidermacs/pull/89
* Doc : fix uv aider proxy error by @ymfsing in https://github.com/MatthewZMD/aidermacs/pull/90
* Fix model inheritance to dynamically use default model values by @u-yuta in https://github.com/MatthewZMD/aidermacs/pull/91
* Handle errors in insert hooks by @CeleritasCelery in https://github.com/MatthewZMD/aidermacs/pull/95
* readme: add spacemacs example by @paralin in https://github.com/MatthewZMD/aidermacs/pull/100
* readme: add back steps 4 and 5 to getting started by @paralin in https://github.com/MatthewZMD/aidermacs/pull/101

## New Contributors
* @manateelazycat made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/89
* @ymfsing made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/90
* @paralin made their first contribution in https://github.com/MatthewZMD/aidermacs/pull/100

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.1...v1.2

# Aidermacs 1.1

**What's New in Aidermacs 1.1?**

* **New Features:**
  * Added common prompts system with history
  * Better support for architect/editor model separation
  * Improved theme support for vterm backend
  * Added model name transformation for API providers

* **Enhanced Model Management:**
  * Added support for weak models for commit messages and chat summarization
  * Improved model selection with prefix arguments for weak model changes
  * Better handling of model inheritance and initialization
  * Added model version checking and caching

* **Output Handling Improvements:**
  * New output module with better file tracking and parsing
  * Improved ediff integration with proper cleanup hooks
  * Better handling of read-only files and remote paths
  * Enhanced output history management

* **Performance Enhancements:**
  * Added protection against infinite font-lock loops
  * Better handling of large output blocks
  * Improved buffer management and cleanup

* **Bug Fixes:**
  * Fixed file path handling in drop commands
  * Improved error handling in model fetching
  * Better handling of process output filtering
  * Fixed issues with buffer naming and selection

**Breaking Changes:**
* Minimum Emacs version requirement changed to 26.1
* Some internal APIs have been reorganized
* Model selection behavior has changed for architect mode

**Upgrade Instructions:**
1. Update your Emacs to at least version 26.1
2. Review your model configuration settings
3. Check for any customizations that may need updating
4. Clear any cached model lists if experiencing issues

**Full Changelog**: https://github.com/MatthewZMD/aidermacs/compare/v1.0...v1.1

# Aidermacs 1.0

**What's New in Aidermacs 1.0?**

Aidermacs 1.0 delivers a range of features designed to enhance your productivity and coding experience:

* **Package Repositories:**
  *   **MELPA:** Install directly from MELPA using `M-x package-install RET aidermacs RET`.
  *   **Non-GNU ELPA:** Also available on Non-GNU ELPA.
*   **Built-in Ediff Integration:** Review AI-generated changes with Emacs' familiar `ediff` interface, allowing you to easily accept or reject modifications.
*   **Intelligent Model Selection:** Automatically discover and integrate with multiple AI providers (OpenAI, Anthropic, DeepSeek, Google Gemini, OpenRouter), ensuring compatibility and optimal performance.
*   **Flexible Terminal Backend Support:** Choose between `comint` and `vterm` backends for the best terminal compatibility and performance.
*   **Enhanced File Management:** Easily manage files within your Aider session with commands for adding, dropping, listing, and more. Full support for remote files via Tramp (SSH, Docker, etc.).
*   **Streamlined Transient Menu Selection:** Access all Aidermacs features through a redesigned and ergonomic transient menu system.
*   **Prompt Files Minor Mode:** Work seamlessly with prompt files and other Aider-related files with convenient keybindings and automatic mode activation.
*   **Claude 3.7 Sonnet Thinking Tokens:** Enable and configure thinking tokens using the `/think-tokens` in-chat command or the `--thinking-tokens` command-line argument.
*   **Architect Mode Confirmation:** Control whether to automatically accept Architect mode changes with the `aidermacs-auto-accept-architect` variable.
*   **Re-Enable Auto-Commits:** Aider automatically commits AI-generated changes by default. We consider this behavior *very* intrusive, so we've disabled it. You can re-enable auto-commits by setting `aidermacs-auto-commits` to `t`.
*   **Customizing Aider Options with `aidermacs-extra-args`:** Pass any Aider-supported command-line options.

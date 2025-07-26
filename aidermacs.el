;;; aidermacs.el --- AI pair programming with Aider -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.5
;; Package-Requires: ((emacs "26.1") (transient "0.3.0") (compat "30.0.2.0") (markdown-mode "2.7"))
;; Keywords: ai emacs llm aider ai-pair-programming tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; SPDX-License-Identifier: Apache-2.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Aidermacs integrates with Aider (https://aider.chat/) for AI-assisted code
;; modification in Emacs. Aider lets you pair program with LLMs to edit code
;; in your local git repository. It works with both new projects and existing
;; code bases, supporting Claude, DeepSeek, ChatGPT, and can connect to almost
;; any LLM including local models. Think of it as having a helpful coding
;; partner that can understand your code, suggest improvements, fix bugs, and
;; even write new code for you. Whether you're working on a new feature,
;; debugging, or just need help understanding some code, Aidermacs provides an
;; intuitive way to collaborate with AI while staying in your familiar Emacs
;; environment.

;; Originally forked from Kang Tu <tninja@gmail.com>'s Aider.el.

;;; Code:

(require 'compat)
(require 'comint)
(require 'dired)
(require 'project)
(require 'transient)
(require 'vc-git)
(require 'which-func)
(require 'ansi-color)
(require 'cl-lib)
(require 'tramp)
(require 'find-dired)

(require 'aidermacs-backends)
(require 'aidermacs-models)
(require 'aidermacs-output)

(declare-function magit-show-commit "magit-diff")

(defgroup aidermacs nil
  "AI pair programming with Aider."
  :group 'aidermacs)

(defcustom aidermacs-program "aider"
  "The name or path of the aidermacs program."
  :type 'string)

(defvar-local aidermacs--current-mode nil
  "Buffer-local variable to track the current aidermacs mode.
Possible values: `code', `ask', `architect', `help'.")

(defvar-local aidermacs--ready nil
  "Buffer-local variable to track whether aider is ready to accept
commands: `nil', `t'")

(defcustom aidermacs-default-chat-mode nil
  "Chat mode used when an Aidermacs session starts.

nil / 'code  – start in code mode (no --chat-mode arg)
'ask         – start in ask/chat mode   (--chat-mode ask)
'architect   – start in architect mode  (--chat-mode architect + model flags)
'help        – start in help mode       (--chat-mode help)."
  :type '(choice (const :tag "Code (default)" nil)
                 (const :tag "Code"       code)
                 (const :tag "Ask"        ask)
                 (const :tag "Architect"  architect)
                 (const :tag "Help"       help)))

;; DEPRECATED – will disappear in a future release
(defcustom aidermacs-use-architect-mode nil
  "OBSOLETE.  Non-nil means start Aidermacs in Architect mode.

Instead, set `aidermacs-default-chat-mode' to the symbol `architect'."
  :type 'boolean
  :group 'aidermacs)

(make-obsolete-variable 'aidermacs-use-architect-mode
                        'aidermacs-default-chat-mode
                        "1.5")

(defcustom aidermacs-config-file nil
  "Path to aider configuration file.
When set, Aidermacs will pass this to aider via --config flag,
ignoring other configuration settings except `aidermacs-extra-args'."
  :type '(choice (const :tag "None" nil)
          (file :tag "Config file")))

(define-obsolete-variable-alias 'aidermacs-args 'aidermacs-extra-args "0.5.0"
  "Old name for `aidermacs-extra-args', please update your config.")

(defcustom aidermacs-extra-args '()
  "Additional arguments to pass to the aidermacs command."
  :type '(repeat string))

(defcustom aidermacs-global-read-only-files '()
  "When non-nil, add read-only files to the aidermacs session. This
is for files that are set not relative to project
directory, ie. project agnostic"
  :type '(repeat string))

(defcustom aidermacs-project-read-only-files '()
  "When non-nil, add read-only files to the aidermacs session. This
is for files that exist relative to the project root."
  :type '(repeat string))

(defcustom aidermacs-subtree-only nil
  "When non-nil, run aider with --subtree-only in the current directory.
This is useful for working in monorepos where you want to limit aider's scope."
  :type 'boolean)

(defcustom aidermacs-auto-commits nil
  "When non-nil, enable auto-commits of LLM changes.
When nil, disable auto-commits requiring manual git commits."
  :type 'boolean)

(defcustom aidermacs-watch-files nil
  "When non-nil, enable watching files for AI coding instructions.
When enabled, aider will watch all files in your repo and look for
any AI coding instructions you add using your favorite IDE or text editor."
  :type 'boolean)

(defcustom aidermacs-auto-accept-architect nil
  "When non-nil, automatically accept architect mode changes.
When nil, require explicit confirmation before applying changes."
  :type 'boolean)

(defcustom aidermacs-exit-kills-buffer nil
  "When non-nil, `aidermacs-exit' will also kill the Aider buffer."
  :type 'boolean)

(defvar aidermacs--read-string-history nil
  "History list for aidermacs read string inputs.")

(defcustom aidermacs-common-prompts
  '("What does this code do? Explain the logic step by step"
    "Explain the overall architecture of this codebase"
    "Simplify this code while preserving functionality"
    "Extract this logic into separate helper functions"
    "Optimize this code for better performance"
    "Are there any edge cases not handled in this code?"
    "Refactor to reduce complexity and improve readability"
    "How could we make this code more maintainable?")
  "List of common prompts to use with aidermacs.
These will be available for selection when using aidermacs commands."
  :type '(repeat string))

(defvar aidermacs--cached-version nil
  "Cached aider version to avoid repeated version checks.")

(defun aidermacs-aider-version ()
  "Check the installed aider version.
Returns a version string like \"0.77.0\" or nil if version can't be determined.
Uses cached version if available to avoid repeated process calls."
  (interactive)
  (let ((path exec-path))
    (or aidermacs--cached-version
        (setq aidermacs--cached-version
              (with-temp-buffer
                (setq-local exec-path path)
                (when (= 0 (process-file aidermacs-program nil t nil "--version"))
                  (goto-char (point-min))
                  (when (re-search-forward "aider \\([0-9]+\\.[0-9]+\\.[0-9]+\\)" nil t)
                    (match-string 1)))))))
  (message "Aider version %s" aidermacs--cached-version)
  aidermacs--cached-version)

(defun aidermacs-clear-aider-version-cache ()
  "Clear the cached aider version.
Call this after upgrading aider to ensure the correct version is detected."
  (interactive)
  (setq aidermacs--cached-version nil)
  (message "Aider version cache cleared."))

(defun aidermacs-project-root ()
  "Get the project root using VC-git, or fallback to file directory.
This function tries multiple methods to determine the project root."
  (or (vc-git-root default-directory)
      (when buffer-file-name
        (file-name-directory buffer-file-name))
      default-directory))

(defcustom aidermacs-prompt-file-name ".aider.prompt.org"
  "File name that will automatically enable `aidermacs-minor-mode' when opened.
This is the file name without path."
  :type 'string)

;;;###autoload (autoload 'aidermacs-transient-menu "aidermacs" nil t)
(transient-define-prefix aidermacs-transient-menu ()
  "AI Pair Programming Interface."
  ["Aidermacs: AI Pair Programming"
   ["Core"
    ("a" "Start/Open Session" aidermacs-run)
    ("." "Start in Current Dir" aidermacs-run-in-current-dir)
    (":" "Start in Chosen Dir" aidermacs-run-in-directory)
    ("l" "Clear Chat History" aidermacs-clear-chat-history)
    ("s" "Reset Session" aidermacs-reset)
    ("x" "Exit Session" aidermacs-exit)]
   ["Persistent Modes"
    ("1" "Code Mode" aidermacs-switch-to-code-mode)
    ("2" "Chat/Ask Mode" aidermacs-switch-to-ask-mode)
    ("3" "Architect Mode" aidermacs-switch-to-architect-mode)
    ("4" "Help Mode" aidermacs-switch-to-help-mode)]
   ["Utilities"
    ("^" "Show Last Commit" aidermacs-magit-show-last-commit
     :if (lambda () aidermacs-auto-commits))
    ("u" "Undo Last Commit" aidermacs-undo-last-commit
     :if (lambda () aidermacs-auto-commits))
    ("C" "Auto-commit Changes" aidermacs-commit-with-auto-message)
    ("R" "Refresh Repo Map" aidermacs-refresh-repo-map)
    ("h" "Session History" aidermacs-show-output-history)
    ("o" "Switch Model (C-u: weak-model)" aidermacs-change-model)
    ("v" "Send Voice" aidermacs-send-voice)
    ("W" "Fetch Web Content" aidermacs-web)
    ("?" "Aider Meta-level Help" aidermacs-help)]]
  ["File Actions"
   ["Add Files (C-u: read-only)"
    ("f" "Add File" aidermacs-add-file)
    ("p" "Add Project File" aidermacs-add-project-file)
    ("F" "Add Current File" aidermacs-add-current-file)
    ("d" "Add From Directory (same type)" aidermacs-add-same-type-files-under-dir)
    ("w" "Add From Window" aidermacs-add-files-in-current-window)
    ("m" "Add From Dired (marked)" aidermacs-batch-add-dired-marked-files)]
   ["Drop Files"
    ("j" "Drop File" aidermacs-drop-file)
    ("J" "Drop Current File" aidermacs-drop-current-file)
    ("k" "Drop From Dired (marked)" aidermacs-batch-drop-dired-marked-files)
    ("K" "Drop All Files" aidermacs-drop-all-files)]
   ["Others"
    ("S" "Create Session Scratchpad" aidermacs-create-session-scratchpad)
    ("G" "Add File to Session" aidermacs-add-file-to-session)
    ("A" "List Added Files" aidermacs-list-added-files)]]
  ["Code Actions"
   ["Code"
    ("c" "Code Change" aidermacs-direct-change)
    ("e" "Question Code" aidermacs-question-code)
    ("r" "Architect Change" aidermacs-architect-this-code)]
   ["Question"
    ("q" "General Question" aidermacs-question-general)
    ("*" "Question This Symbol" aidermacs-question-this-symbol)
    ("g" "Accept Proposed Changes" aidermacs-accept-change)]
   ["Others"
    ("i" "Implement TODO" aidermacs-implement-todo)
    ("t" "Write Test" aidermacs-write-unit-test)
    ("T" "Fix Test" aidermacs-fix-failing-test-under-cursor)
    ("!" "Debug Exception" aidermacs-debug-exception)]])

(defun aidermacs-select-buffer-name ()
  "Select an existing aidermacs session buffer.
If there is only one aidermacs buffer, return its name.
If there are multiple, prompt to select one interactively.
Returns nil if no aidermacs buffers exist.
This is used when you want to target an existing session."
  (let* ((buffers (match-buffers #'aidermacs--is-aidermacs-buffer-p))
         (buffer-names (mapcar #'buffer-name buffers)))
    (pcase buffers
      (`() nil)
      (`(,name) (buffer-name name))
      (_ (completing-read "Select aidermacs session: " buffer-names nil t)))))

(defun aidermacs-get-buffer-name (&optional use-existing suffix)
  "Generate the aidermacs buffer name based on project root or current directory.
If USE-EXISTING is non-nil, use an existing buffer instead of creating new.
If supplied, SUFFIX is appended to the buffer name within the earmuffs."
  (if use-existing
      (aidermacs-select-buffer-name)
    (let* ((root (aidermacs-project-root))
           (current-dir-truename (file-truename default-directory)) ; Use truename for consistent comparisons
           (aidermacs-buffers (match-buffers #'aidermacs--is-aidermacs-buffer-p))
           ;; Extract truename paths from existing *aidermacs:PATH* buffers (base sessions)
           (existing-session-paths
            (delq nil
                  (mapcar (lambda (buf)
                            (when (string-match "^\\*aidermacs:\\([^\\*]+\\)\\*$" (buffer-name buf)) ; Match base session names
                              (let ((path-str (match-string 1 (buffer-name buf))))
                                (when (file-directory-p path-str) ; Ensure it's a directory
                                  (file-truename path-str)))))
                          aidermacs-buffers)))

           ;; Determine the display-root for the buffer name
           (display-root
            (let* ((sessions-containing-current-dir ;; Sessions whose paths contain current-dir-truename
                    (sort (cl-remove-if-not
                           (lambda (session-path)
                             (file-in-directory-p current-dir-truename session-path))
                           existing-session-paths)
                          ;; Sort by length (deeper paths are more specific)
                          (lambda (a b) (> (length a) (length b)))))
                   (closest-session-containing-current-dir (car sessions-containing-current-dir))

                   (sessions-within-current-dir ;; Sessions whose paths are within current-dir-truename
                    (sort (cl-remove-if-not
                           (lambda (session-path)
                             (file-in-directory-p session-path current-dir-truename))
                           existing-session-paths)
                          ;; Sort by length (deeper paths are more specific)
                          (lambda (a b) (> (length a) (length b))))))

              (cond
               ;; 1. Current directory is INSIDE an existing session's directory.
               (closest-session-containing-current-dir
                closest-session-containing-current-dir)

               ;; 2. Current directory is an ANCESTOR of exactly ONE existing session.
               ((and (not closest-session-containing-current-dir) ; Only if not already covered by case 1
                     sessions-within-current-dir
                     (= 1 (length sessions-within-current-dir)))
                (car sessions-within-current-dir))

               ;; 3. Fallback logic (original logic for new session or ambiguous cases).
               (aidermacs-subtree-only current-dir-truename)
               (t (file-truename root)))))) ; Ensure root is also truename for consistency

      (format "*aidermacs:%s%s*"
              display-root
              (or suffix "")))))

(defun aidermacs--live-p (buffer-name)
  "Return t if the aider buffer is availble and process is currently running"
  (and (get-buffer buffer-name)
       (process-live-p (get-buffer-process (or buffer-name (aidermacs-get-buffer-name))))))

;;;###autoload
(defun aidermacs-run ()
  "Run aidermacs process using the selected backend.
This function sets up the appropriate arguments and launches the process."
  (interactive)
  ;; Legacy boolean ⇒ new symbol
  (when (and aidermacs-use-architect-mode
             (null aidermacs-default-chat-mode))
    (setq aidermacs-default-chat-mode 'architect)
    (display-warning
     'aidermacs
     "`aidermacs-use-architect-mode' is obsolete – \
set `aidermacs-default-chat-mode' to 'architect' instead."
     :warning))
  ;; Set up necessary hooks when aidermacs is actually run
  (aidermacs--setup-ediff-cleanup-hooks)
  (aidermacs--setup-cleanup-hooks)
  (aidermacs-setup-minor-mode)

  (let* ((buffer-name (aidermacs-get-buffer-name))
         ;; Split each string on whitespace for member comparison later
         (flat-extra-args
          (cl-mapcan (lambda (s)
                       (if (stringp s)
                           (split-string s "[[:space:]]+" t)
                         (list s)))
                     aidermacs-extra-args))
         (has-model-arg (cl-some (lambda (x) (member x flat-extra-args))
                                 '("--model" "--opus" "--sonnet" "--haiku"
                                   "--4" "--4o" "--mini" "--4-turbo" "--35turbo"
                                   "--deepseek" "--o1-mini" "--o1-preview")))
         (has-config-arg (or (cl-some (lambda (dir)
                                        (let ((conf (expand-file-name ".aider.conf.yml" dir)))
                                          (when (file-exists-p conf)
                                            dir)))
                                      (list (expand-file-name "~")
                                            (aidermacs-project-root)
                                            default-directory))
                             aidermacs-config-file
                             (cl-some (lambda (x) (member x flat-extra-args))
                                      '("--config" "-c"))))
         ;; Check aider version for auto-accept-architect support
         (aider-version (aidermacs-aider-version))

         ;; New code: determine startup mode and warnings
         (startup-mode (or aidermacs-default-chat-mode 'code))
         (needs-chat-flag (and startup-mode (not (eq startup-mode 'code))))
         (architect? (eq startup-mode 'architect)))
    (let* ((backend-args
            (if has-config-arg
                ;; Only need to add aidermacs-config-file manually
                (when aidermacs-config-file
                  (list "--config" aidermacs-config-file))
              (append
               ;; --chat-mode when required (ask/help/architect)
               (when needs-chat-flag
                 (list "--chat-mode" (symbol-name startup-mode)))

               ;; extra model flags for architect
               (when architect?
                 (list "--model"        (aidermacs-get-architect-model)
                       "--editor-model" (aidermacs-get-editor-model)))

               ;; default --model when we’re still in plain code mode and user
               ;; didn’t supply one
               (when (and (not needs-chat-flag) (not has-model-arg))
                 (list "--model" aidermacs-default-model))

               ;; existing flags that follow (no change)
               (unless aidermacs-auto-commits
                 '("--no-auto-commits"))
               ;; Only add --no-auto-accept-architect if:
               ;; 1. User has disabled auto-accept (aidermacs-auto-accept-architect is nil)
               ;; 2. Aider version supports this flag (>= 0.77.0)
               (when (and (not aidermacs-auto-accept-architect)
                          (version<= "0.77.0" aider-version))
                 '("--no-auto-accept-architect"))
               ;; Add watch-files if enabled
               (when aidermacs-watch-files
                 '("--watch-files"))
               ;; Add weak model if specified
               (when aidermacs-weak-model
                 (list "--weak-model" aidermacs-weak-model))
               (when aidermacs-subtree-only
                 '("--subtree-only"))
               (when aidermacs-global-read-only-files
                 (apply #'append
                        (mapcar (lambda (file) (list "--read" file))
                                aidermacs-global-read-only-files)))
               (when aidermacs-project-read-only-files
                 (apply #'append
                        (mapcar (lambda (file) (list "--read"
                                                     (expand-file-name file (aidermacs-project-root))))
                                aidermacs-project-read-only-files))))))
           ;; Take the original aidermacs-extra-args instead of the flat ones
           (final-args (append backend-args aidermacs-extra-args)))
      (if (aidermacs--live-p buffer-name)
          (aidermacs-switch-to-buffer buffer-name)
        (aidermacs-run-backend aidermacs-program final-args buffer-name)
        (with-current-buffer buffer-name
          ;; Set initial mode based on startup configuration
          (setq-local aidermacs--current-mode startup-mode))
        (aidermacs-switch-to-buffer buffer-name)))))

(defun aidermacs-run-in-current-dir ()
  "Run aidermacs in the current directory with --subtree-only flag.
This is useful for working in monorepos where you want to limit aider's scope."
  (interactive)
  (let ((aidermacs-subtree-only t))
    (aidermacs-run)))

(defun aidermacs-run-in-directory (directory)
  "Prompt for a directory and run aidermacs with --subtree-only flag.
This is useful for working in complex monorepos with nested subprojects."
  (interactive
   (list (read-file-name "Choose a directory: " nil nil t nil 'file-directory-p)))
   (let ((aidermacs-subtree-only t)
         (default-directory directory))
    (aidermacs-run)))

(defun aidermacs--command-may-edit-files (command)
  "Check if COMMAND may result in file edits.
Returns t if the command is likely to modify files, nil otherwise.
In code/architect mode, commands without prefixes may edit.
Commands containing /code or /architect always may edit."
  (and (stringp command)
       (or (and (memq aidermacs--current-mode '(code architect))
                (not (string-match-p "^/" command)))
           (string-match-p "/code" command)
           (string-match-p "/architect" command))))

(defun aidermacs--send-command (command &optional no-switch-to-buffer use-existing redirect callback)
  "Send command to the corresponding aidermacs process.
COMMAND is the text to send.
If NO-SWITCH-TO-BUFFER is non-nil, don't switch to the aidermacs buffer.
If USE-EXISTING is non-nil, use an existing buffer instead of creating new.
If REDIRECT is non-nil it redirects the output (hidden) for comint backend.
If CALLBACK is non-nil it will be called after the command finishes."
  (let* ((buffer-name (aidermacs-get-buffer-name use-existing))
         (buffer (if (aidermacs--live-p buffer-name)
                     (get-buffer buffer-name)
                   (when (get-buffer buffer-name)
                     (kill-buffer buffer-name))
                   (aidermacs-run)
                   (sit-for 1)
                   (get-buffer buffer-name)))
         (processed-command (aidermacs--process-message-if-multi-line command)))
    ;; Check if command may edit files and prepare accordingly
    (with-current-buffer buffer
      ;; Attempt to wait out transient commands or server lag
      (when (not aidermacs--ready)
        (sit-for 0.5))
      (if (not aidermacs--ready)
          (progn (aidermacs-switch-to-buffer buffer-name)
                 (message "Aider process is not currently accepting commands"))
        ;; Reset current output before sending new command
        (setq aidermacs--current-output "")
        (setq aidermacs--current-callback callback)
        (setq aidermacs--last-command processed-command)
        (aidermacs--cleanup-temp-buffers)
        (aidermacs--ensure-current-file-tracked)
        (when (aidermacs--command-may-edit-files command)
          (aidermacs--prepare-for-code-edit))
        (aidermacs--send-command-backend buffer processed-command redirect)))
    (when (and (not no-switch-to-buffer)
               (not (string= (buffer-name) buffer-name)))
      (aidermacs-switch-to-buffer buffer-name))))

;;;###autoload
(defun aidermacs-switch-to-buffer (&optional buffer-name)
  "Switch to the aidermacs buffer.
If BUFFER-NAME is provided, switch to that buffer.
If not, try to get a buffer using `aidermacs-get-buffer-name`.
If that fails, try an existing buffer with `aidermacs-select-buffer-name`.
If the buffer is already visible in a window, switch to that window.
If the current buffer is already the aidermacs buffer, do nothing."
  (interactive)
  (let* ((target-buffer-name (or buffer-name
                                 (aidermacs-get-buffer-name t)
                                 (aidermacs-select-buffer-name)))
         (buffer (and target-buffer-name (get-buffer target-buffer-name))))
    (cond
     ((and target-buffer-name (string= (buffer-name) target-buffer-name)) t)
     ((and buffer (get-buffer-window buffer))
      (select-window (get-buffer-window buffer)))  ;; Switch to existing window
     (buffer
      (pop-to-buffer buffer))
     (t
      (error "No aidermacs buffer exists")))))

(defun aidermacs-clear-chat-history ()
  "Send the command \"/clear\" to the aidermacs buffer."
  (interactive)
  (aidermacs--send-command "/clear"))

(defun aidermacs-reset ()
  "Send the command \"/reset\" to the aidermacs buffer."
  (interactive)
  (setq aidermacs--tracked-files nil)
  (aidermacs--send-command "/reset"))

(defun aidermacs-exit ()
  "Send the command \"/exit\" to the aidermacs buffer.
If `aidermacs-exit-kills-buffer' is non-nil, also kill the buffer."
  (interactive)
  (let ((buffer-name (aidermacs-get-buffer-name)))
    (when (get-buffer buffer-name)
      (aidermacs--cleanup-temp-buffers)
      (when (aidermacs--live-p buffer-name)
        (aidermacs--send-command "/exit" t))
      (when aidermacs-exit-kills-buffer
        (sit-for 1)
        (kill-buffer buffer-name)))))

(defun aidermacs--process-message-if-multi-line (str)
  "Process multi-line chat messages for proper formatting.
STR is the message to process.  If STR contains newlines and isn't already
wrapped in {aidermacs...aidermacs}, wrap it.
Otherwise return STR unchanged.  See documentation at:
https://aidermacs.chat/docs/usage/commands.html#entering-multi-line-chat-messages"
  (if (and (string-match-p "\n" str)
           (not (string-match-p "^{aidermacs\n.*\naidermacs}$" str)))
      (format "{aidermacs\n%s\naidermacs}" str)
    str))

(defun aidermacs-drop-current-file ()
  "Drop the current file from aidermacs session."
  (interactive)
  (if (not buffer-file-name)
      (message "Current buffer is not associated with a file.")
    (let* ((file-path (aidermacs--localize-tramp-path buffer-file-name))
           (formatted-path (if (string-match-p " " file-path)
                               (format "\"%s\"" file-path)
                             file-path))
           (command (format "/drop %s" formatted-path)))
      (aidermacs--send-command command))))

(defun aidermacs--parse-ls-output (output)
  "Parse the /ls command output to extract files in chat.
OUTPUT is the text returned by the /ls command.  After the \"Files in chat:\"
header, each subsequent line that begins with whitespace is processed.
The first non-whitespace token is taken as the file name.  Relative paths are
resolved using the repository root (if available) or `default-directory`.
Only files that exist on disk are included in the result.
Returns a deduplicated list of such file names."
  (when output
    (with-temp-buffer
      (insert output)
      (goto-char (point-min))
      (let* ((files '())
             (base (aidermacs-project-root))
             (is-remote (file-remote-p base)))
        ;; Parse read-only files section
        (when (search-forward "Read-only files:" nil t)
          (forward-line 1)
          (while (and (not (eobp))
                      (string-match-p "^[[:space:]]" (thing-at-point 'line t)))
            (let* ((line (string-trim (thing-at-point 'line t)))
                   (file (car (split-string line))))
              ;; For remote files, we don't try to verify existence or convert paths
              (when file
                (if is-remote
                    (push (concat file " (read-only)") files)
                  ;; For local files, verify existence and convert to relative path
                  (when (file-exists-p (expand-file-name file base))
                    (push (concat (file-relative-name (expand-file-name file base) base)
                                  " (read-only)")
                          files)))))
            (forward-line 1)))

        ;; Parse files in chat section
        (when (search-forward "Files in chat:" nil t)
          (forward-line 1)
          (while (and (not (eobp))
                      (string-match-p "^[[:space:]]" (thing-at-point 'line t)))
            (let* ((line (string-trim (thing-at-point 'line t)))
                   (file (car (split-string line))))
              ;; For remote files, we don't try to verify existence or convert paths
              (when file
                (if is-remote
                    (push file files)
                  ;; For local files, verify existence and convert to relative path
                  (when (file-exists-p (expand-file-name file base))
                    (push (file-relative-name (expand-file-name file base) base) files)))))
            (forward-line 1)))

        ;; Remove duplicates and return
        (setq aidermacs--tracked-files (delete-dups (nreverse files)))
        aidermacs--tracked-files))))

(defun aidermacs--get-files-in-session (callback)
  "Get list of files in current session and call CALLBACK with the result."
  (aidermacs--send-command
   "/ls" nil nil t
   (lambda ()
     (let ((files (aidermacs--parse-ls-output aidermacs--current-output)))
       (funcall callback files)))))

(defun aidermacs-list-added-files ()
  "List all files currently added to the chat session.
Sends the \"/ls\" command and displays the results in a Dired buffer."
  (interactive)
  (aidermacs--get-files-in-session
   (lambda (files)
     (setq aidermacs--tracked-files files)
     (let ((buf-name (aidermacs-get-buffer-name nil " Files")))
       ;; Unfortunately find-dired-with-command doesn't allow us to specify the
       ;; buffer name, so we manually rename it after the fact and recreate it
       ;; on each call.
       (when (get-buffer buf-name)
         (kill-buffer buf-name))
       (if files
           (let* ((root (aidermacs-project-root))
                  (files-arg (mapconcat #'shell-quote-argument files " "))
                  (cmd (format "find %s %s" files-arg (car find-ls-option))))
             (find-dired-with-command root cmd)
             (let ((buf (get-buffer "*Find*")))
               (when buf
                 (with-current-buffer buf
                   (rename-buffer buf-name)
                   (save-excursion
                     ;; The executed command is on the 2nd line; it can get
                     ;; quite long, so we delete it to avoid cluttering the
                     ;; buffer.
                     (goto-char (point-min))
                     (forward-line 1)  ;; Move to the 2nd line
                     (when (looking-at "^ *find " t)
                       (let ((inhibit-read-only t))
                         (delete-region (line-beginning-position) (line-end-position)))))
                   (setq revert-buffer-function
                         (lambda (&rest _) (aidermacs-list-added-files)))))))
         (message "No files added to the chat session"))))))

(defun aidermacs-drop-file ()
  "Drop a file from the chat session by selecting from currently added files."
  (interactive)
  (aidermacs--get-files-in-session
   (lambda (files)
     (if-let* ((file (completing-read "Select file to drop: " files nil t))
               (clean-file (replace-regexp-in-string " (read-only)$" "" file)))
         (let ((command (aidermacs--prepare-file-paths-for-command "/drop" (list clean-file))))
           (aidermacs--send-command command))
       (message "No files available to drop")))))

(defun aidermacs-drop-all-files ()
  "Drop all files from the current chat session."
  (interactive)
  (setq aidermacs--tracked-files nil)
  (aidermacs--send-command "/drop"))

(defun aidermacs-batch-drop-dired-marked-files ()
  "Drop Dired marked files from the aidermacs session."
  (interactive)
  (unless (derived-mode-p 'dired-mode)
    (user-error "This command can only be used in Dired mode"))
  (let ((files (dired-get-marked-files))
        (is-aidermacs-files-buffer (string= (buffer-name)
                                            (aidermacs-get-buffer-name nil " Files"))))
    (aidermacs--drop-files-helper files)
    ;; If we're in the special aidermacs files buffer, kill it after dropping files
    (when is-aidermacs-files-buffer
      (message "Closing aidermacs file buffer after dropping files")
      (kill-buffer (aidermacs-get-buffer-name nil " Files")))))

(defun aidermacs--form-prompt (command &optional prompt-prefix guide ignore-context)
  "Get command based on context with COMMAND and PROMPT-PREFIX.
COMMAND is the text to prepend.  PROMPT-PREFIX is the text to add after COMMAND.
GUIDE is displayed in the prompt but not included in the final command.
Use highlighted region as context unless IGNORE-CONTEXT is set to non-nil."
  (let* ((region-text (when (and (use-region-p) (not ignore-context))
                        (buffer-substring-no-properties (region-beginning) (region-end))))
         (context (when region-text
                    (format " in %s regarding this section:\n```\n%s\n```\n" (buffer-name) region-text)))
         ;; Read user input
         (user-command
          (read-string
           (concat command " " prompt-prefix context
                   (when guide (format " (%s)" guide)) ": ")
           nil 'aidermacs--read-string-history nil nil)))
    ;; Add to history if not already there, removing any duplicates
    (setq aidermacs--read-string-history
          (delete-dups (cons user-command aidermacs--read-string-history)))
    (concat command
            " "
            prompt-prefix
            context
            (unless (string-empty-p user-command)
              (concat ": " user-command)))))

(defun aidermacs-direct-change ()
  "Prompt the user for an input and send it to aidermacs prefixed with \"/code \"."
  (interactive)
  (when-let* ((command (aidermacs--form-prompt "/code" "Make this change" "will edit file")))
    (aidermacs--ensure-current-file-tracked)
    (aidermacs--send-command command)))

(defun aidermacs-question-code ()
  "Ask a question about the code at point or region.
If a region is active, include the region text in the question.
If cursor is inside a function, include the function name as context.
If called from the aidermacs buffer, use general question instead."
  (interactive)
  (when-let* ((command (aidermacs--form-prompt "/ask" "Propose a solution" "won't edit file")))
    (aidermacs--ensure-current-file-tracked)
    (aidermacs--send-command command)))

(defun aidermacs-architect-this-code ()
  "Architect code at point or region.
If region is active, inspect that region.
If point is in a function, inspect that function."
  (interactive)
  (when-let* ((command (aidermacs--form-prompt "/architect" "Design a solution" "confirm before edit")))
    (aidermacs--ensure-current-file-tracked)
    (aidermacs--send-command command)))

(defun aidermacs-question-general ()
  "Prompt the user for a general question without code context."
  (interactive)
  (when-let* ((command (aidermacs--form-prompt "/ask" nil "empty for ask mode" t)))
    (aidermacs--send-command command)))

(defun aidermacs-help ()
  "Prompt the user for an input prefixed with \"/help \"."
  (interactive)
  (when-let* ((command (aidermacs--form-prompt "/help" nil "question how to use aider, empty for all commands" t)))
    (aidermacs--send-command command)))

(defun aidermacs-debug-exception ()
  "Prompt the user for an input and send it to aidermacs prefixed with \"/debug \"."
  (interactive)
  (when-let* ((command (aidermacs--form-prompt "/ask" "Debug exception")))
    (aidermacs--send-command command)))

(defun aidermacs-accept-change ()
  "Send the command \"go ahead\" to the aidermacs."
  (interactive)
  (aidermacs--send-command "/code ok"))

(defun aidermacs-magit-show-last-commit ()
  "Show the last commit message using Magit.
If Magit is not installed, report that it is required."
  (interactive)
  (if (require 'magit nil 'noerror)
      (magit-show-commit "HEAD")
    (message "Magit is required to show the last commit.")))

(defun aidermacs-undo-last-commit ()
  "Undo the last change made by aidermacs."
  (interactive)
  (aidermacs--send-command "/undo"))

(defun aidermacs-commit-with-auto-message ()
  "Commit edits to the repo with an automatically generated commit message.
Uses aider's /commit command without arguments to generate a descriptive
commit message automatically based on the changes made."
  (interactive)
  (aidermacs--send-command "/commit"))

(defun aidermacs-question-this-symbol ()
  "Ask aidermacs to explain symbol under point."
  (interactive)
  (let* ((symbol (thing-at-point 'symbol))
         (line (string-trim-right (thing-at-point 'line)))
         (prompt (format "/ask Please explain what '%s' means in the context of this code line: %s"
                         symbol line)))
    (unless symbol
      (error "No symbol under point!"))
    (aidermacs--ensure-current-file-tracked)
    (aidermacs--send-command prompt)))

(defun aidermacs-send-command-with-prefix (prefix command)
  "Send COMMAND to the aidermacs buffer with PREFIX.
PREFIX is the text to prepend.  COMMAND is the text to send."
  (aidermacs--ensure-current-file-tracked)
  (aidermacs--send-command (concat prefix command)))

(defun aidermacs--localize-tramp-path (file)
  "If FILE is a TRAMP path, extract the local part of the path.
Otherwise, return FILE unchanged."
  (if (and (fboundp 'tramp-tramp-file-p) (tramp-tramp-file-p file))
      (let ((local-name (tramp-file-name-localname (tramp-dissect-file-name file))))
        local-name)
    file))

(defun aidermacs--prepare-file-paths-for-command (command files)
  "Prepare FILES for use with COMMAND in aider.
Handles TRAMP paths by extracting local parts and formats the command string,
but wrapping them with double quotes that aider understands."
  (let* ((localized-files (mapcar #'aidermacs--localize-tramp-path (delq nil files)))
         (quoted-files (mapcar (lambda (path) (format "\"%s\"" path)) localized-files)))
    (if quoted-files
        (format "%s %s" command
                (mapconcat #'identity quoted-files " "))
      (format "%s" command))))

(defun aidermacs--add-files-helper (files &optional read-only message)
  "Helper function to add files with read-only flag.
FILES is a list of file paths to add. READ-ONLY determines if files are added
as read-only.  MESSAGE can override the default success message."
  (let* ((cmd (if read-only "/read-only" "/add"))
         (command (aidermacs--prepare-file-paths-for-command cmd files))
         (files (delq nil files)))
    (if files
        (progn
          (aidermacs--send-command command)
          (message (or message
                       (format "Added %d files as %s"
                               (length files)
                               (if read-only "read-only" "editable")))))
      (message "No files to add."))))

(defun aidermacs--drop-files-helper (files &optional message)
  "Helper function to drop files.
FILES is a list of file paths to drop.  Optional MESSAGE can override the
default success message."
  (let* ((command (aidermacs--prepare-file-paths-for-command "/drop" files))
         (files (delq nil files)))
    (if files
        (progn
          (aidermacs--send-command command)
          (message (or message
                       (format "Dropped %d files"
                               (length files)))))
      (message "No files to drop."))))

(defun aidermacs-add-current-file (&optional read-only)
  "Add current file with optional READ-ONLY flag.
With prefix argument `C-u', add as read-only."
  (interactive "P")
  (aidermacs--add-files-helper
   (if buffer-file-name (list buffer-file-name) nil)
   read-only
   (when buffer-file-name
     (format "Added %s as %s"
             (file-name-nondirectory buffer-file-name)
             (if read-only "read-only" "editable")))))

(defun aidermacs--pick-project-file ()
  "Prompt for a file in the current project using `completing-read`.
This function attempts to use `project.el` to find files if available
and consistent with `aidermacs-project-root`. Otherwise, it falls back
to a recursive directory listing based on `aidermacs-project-root`."
  (interactive)
  (let* ((aidermacs-root-raw (aidermacs-project-root))
         (aidermacs-root (when aidermacs-root-raw (expand-file-name aidermacs-root-raw)))
         (project-files-list nil)
         (base-for-relativization aidermacs-root))

    (unless aidermacs-root
      (user-error "No project root found by aidermacs-project-root"))

    (if (and (fboundp 'project-files) (project-current))
        (let* ((proj (project-current)) ; proj is (TYPE . DIR) or (TYPE . (Git "DIR"))
               (project-el-root-candidate (cdr proj)) ; This can be DIR string or a list like (Git "DIR")
               (project-el-root-str
                (cond
                 ((stringp project-el-root-candidate) project-el-root-candidate)
                 ;; Handles (SYMBOL "path" ...) e.g. (Git "/path/to/root")
                 ((and (consp project-el-root-candidate)
                       (symbolp (car project-el-root-candidate))
                       (>= (length project-el-root-candidate) 2)
                       (stringp (nth 1 project-el-root-candidate)))
                  (nth 1 project-el-root-candidate))
                 (t nil)))
               (project-el-root-expanded (when project-el-root-str (expand-file-name project-el-root-str))))

          (cond
           ((not project-el-root-expanded)
            (message "aidermacs--pick-project-file: Could not determine project.el root string. Falling back to recursive listing based on aidermacs-project-root."))
           ((string= aidermacs-root project-el-root-expanded)
            (setq project-files-list (project-files proj))
            (setq base-for-relativization project-el-root-expanded)
            (unless project-files-list
              (message "aidermacs--pick-project-file: project-files for '%s' returned no files. Falling back." project-el-root-expanded)))
           (t
            (message "aidermacs--pick-project-file: aidermacs-project-root (%s) differs from project.el root (%s). Falling back to recursive listing based on aidermacs-project-root."
                     aidermacs-root project-el-root-expanded))))
      (message "aidermacs--pick-project-file: project.el not available or no current project. Falling back to recursive listing based on aidermacs-project-root."))

    ;; Fallback if project-files were not used or yielded no files
    (unless project-files-list
      (setq project-files-list (directory-files-recursively aidermacs-root ".*" t))
      (setq base-for-relativization aidermacs-root)) ; Ensure base is set for fallback

    (unless project-files-list
      (user-error "No files found in project: %s" base-for-relativization))

    (let* ((choices (mapcar (lambda (f) (file-relative-name f base-for-relativization))
                            project-files-list))
           (selected-relative-file (completing-read "Select project file to add: " choices nil t)))
      (when selected-relative-file
        (expand-file-name selected-relative-file base-for-relativization)))))

(defun aidermacs-add-file (&optional read-only)
  "Add file(s) to aidermacs interactively.
With prefix argument `C-u', add as READ-ONLY.
If current buffer is visiting a file, its name is used as initial input.
Multiple files can be selected by calling the command multiple times."
  (interactive "P")
  (let ((file (expand-file-name
               (read-file-name "Select file to add: "
                               nil nil t))))
    (cond
     ((file-directory-p file)
      (when (yes-or-no-p (format "Add all files in directory %s? " file))
        (aidermacs--add-files-helper
         (directory-files file t "^[^.]" t)  ;; Exclude dotfiles
         read-only
         (format "Added all files in %s as %s"
                 file (if read-only "read-only" "editable")))))
     ((file-exists-p file)
      (aidermacs--add-files-helper
       (list file)
       read-only
       (format "Added %s as %s"
               (file-name-nondirectory file)
               (if read-only "read-only" "editable")))))))

(defun aidermacs-add-project-file (&optional read-only)
  "Add a file from the current project to the aider session.
With prefix argument `C-u', add as READ-ONLY."
  (interactive "P")
  (let ((file (aidermacs--pick-project-file)))
    (aidermacs--add-files-helper
     (list file)
     read-only
     (format "Added %s from project as %s"
             (file-name-nondirectory file)
             (if read-only "read-only" "editable")))))

(defun aidermacs-add-files-in-current-window (&optional read-only)
  "Add window files with READ-ONLY flag.
With prefix argument `C-u', add as read-only."
  (interactive "P")
  (let* ((files (mapcan (lambda (window)
                          (with-current-buffer (window-buffer window)
                            (and buffer-file-name
                                 (list (expand-file-name buffer-file-name)))))
                        (window-list))))
    (aidermacs--add-files-helper files read-only)))

(defun aidermacs-batch-add-dired-marked-files (&optional read-only)
  "Add Dired files with READ-ONLY flag.
With prefix argument `C-u', add as read-only."
  (interactive "P")
  (unless (derived-mode-p 'dired-mode)
    (user-error "This command can only be used in Dired mode"))
  (aidermacs--add-files-helper (dired-get-marked-files) read-only))

(defun aidermacs-add-same-type-files-under-dir (&optional read-only)
  "Add all files with same suffix as current file under current directory.
If there are more than 40 files, refuse to add and show warning message.
With prefix argument `C-u', add as READ-ONLY."
  (interactive "P")
  (if (not buffer-file-name)
      (user-error "Current buffer is not visiting a file")
    (let* ((current-suffix (file-name-extension buffer-file-name))
           (dir (file-name-directory buffer-file-name))
           (max-files 40)
           (files (directory-files dir t (concat "\\." current-suffix "$") t)))
      (if (length> files max-files)
          (message "Too many files (%d, > %d) found with suffix .%s. Aborting."
                   (length files) max-files current-suffix)
        (aidermacs--add-files-helper files read-only
                                     (format "Added %d files with suffix .%s as %s"
                                             (length files) current-suffix
                                             (if read-only "read-only" "editable")))))))

;;;###autoload
(defun aidermacs-write-unit-test ()
  "Generate unit test code for current buffer.
Do nothing if current buffer is not visiting a file.
If current buffer filename contains `test':
  - If cursor is inside a test function, implement that test
  - Otherwise show message asking to place cursor inside a test function
Otherwise:
  - If cursor is on a function, generate unit test for that function
  - Otherwise generate unit tests for the entire file"
  (interactive)
  (if (not buffer-file-name)
      (user-error "Current buffer is not visiting a file")
    (let ((function-name (which-function)))
      (cond
       ;; Test file case
       ((string-match-p "test" (file-name-nondirectory buffer-file-name))
        (if function-name
            (if (string-match-p "test" function-name)
                (let* ((initial-input
                        (format "Please implement test function '%s'. Follow standard unit testing practices and make it a meaningful test. Do not use Mock if possible."
                                function-name))
                       (command (aidermacs--form-prompt "/architect" initial-input)))
                  (aidermacs--ensure-current-file-tracked)
                  (aidermacs--send-command command))
              (message "Current function '%s' does not appear to be a test function." function-name))
          (message "Please place cursor inside a test function to implement.")))
       ;; Non-test file case
       (t
        (let* ((common-instructions "Keep existing tests if there are. Follow standard unit testing practices. Do not use Mock if possible.")
               (initial-input
                (if function-name
                    (format "Please write unit test code for function '%s'. %s"
                            function-name common-instructions)
                  (format "Please write unit test code for file '%s'. For each function %s"
                          (file-name-nondirectory buffer-file-name) common-instructions)))
               (command (aidermacs--form-prompt "/architect" initial-input)))
          (aidermacs--ensure-current-file-tracked)
          (aidermacs--send-command command)))))))

;;;###autoload
(defun aidermacs-fix-failing-test-under-cursor ()
  "Report the current test failure to aidermacs and ask it to fix the code.
This function assumes the cursor is on or inside a test function."
  (interactive)
  (if-let* ((test-function-name (which-function)))
      (let* ((initial-input (format "The test '%s' is failing. Please analyze and fix the code to make the test pass. Don't break any other test"
                                    test-function-name))
             (command (aidermacs--form-prompt "/architect" initial-input)))
        (aidermacs--ensure-current-file-tracked)
        (aidermacs--send-command command))
    (message "No test function found at cursor position.")))

(defun aidermacs-create-session-scratchpad ()
  "Create a new temporary file for adding content to the aider session.
The file will be created in the system's temp directory
with a timestamped name.  Use this to add functions, code
snippets, or other content to the session."
  (interactive)
  (let* ((temp-dir (file-name-as-directory (temporary-file-directory)))
         (filename (expand-file-name
                    (format "aidermacs-%s.txt" (format-time-string "%Y%m%d-%H%M%S"))
                    temp-dir)))
    ;; Create and populate the file safely
    (with-temp-buffer
      (insert ";; Temporary scratchpad created by aidermacs\n")
      (insert ";; Add your code snippets, functions, or other content here\n")
      (insert ";; Just edit and save - changes will be available to aider\n\n")
      (write-file filename))
    (let ((command (aidermacs--prepare-file-paths-for-command "/read" (list filename))))
      (aidermacs--send-command command t t))
    (find-file-other-window filename)
    (message "Created and added scratchpad to session: %s" filename)))

(defun aidermacs-add-file-to-session (&optional file)
  "Interactively add a FILE to an existing aidermacs session using /read.
This allows you to add the file's content to a specific session."
  (interactive
   (let* ((initial (when buffer-file-name
                     (file-name-nondirectory buffer-file-name))))
     (list (read-file-name "Select file to add to existing session: "
                           nil nil t initial))))
  (cond
   ((not (file-exists-p file))
    (error "File does not exist: %s" file))
   ((file-directory-p file)
    (when (yes-or-no-p (format "Add all files in directory %s? " file))
      (let ((command (aidermacs--prepare-file-paths-for-command
                      "/read"
                      (directory-files file t "^[^.]" t))))  ;; Exclude dotfiles
        (aidermacs--send-command command nil t)
        (message "Added all files in %s to session" file))))
   (t (let ((command (aidermacs--prepare-file-paths-for-command "/read" (list file))))
        (aidermacs--send-command command nil t)
        (message "Added %s to session" (file-name-nondirectory file))))))

(defun aidermacs--is-comment-line (line)
  "Check if LINE is a comment line based on current buffer's comment syntax.
Returns non-nil if LINE starts with one or more
comment characters, ignoring leading whitespace."
  (when comment-start
    (let ((comment-str (string-trim-right comment-start)))
      (string-match-p (concat "^[ \t]*"
                              (regexp-quote comment-str)
                              "+")
                      (string-trim-left line)))))

;;;###autoload
(defun aidermacs-implement-todo ()
  "Implement TODO comments in current context.
If region is active, implement that specific region.
If cursor is on a comment line, implement that specific comment.
If point is in a function, implement TODOs for that function.
Otherwise implement TODOs for the entire current file."
  (interactive)
  (if (not buffer-file-name)
      (message "Current buffer is not visiting a file.")
    (let* ((current-line (string-trim (thing-at-point 'line t)))
           (is-comment (aidermacs--is-comment-line current-line)))
      (when-let* ((command (aidermacs--form-prompt
                            "/architect"
                            (concat "Please implement the TODO items."
                                    (and is-comment
                                         (format " on this comment: `%s`." current-line))
                                    " Keep existing code structure"))))
        (aidermacs--ensure-current-file-tracked)
        (aidermacs--send-command command)))))

(defun aidermacs-send-line-or-region ()
  "Send text to the aidermacs buffer.
If region is active, send the selected region.
Otherwise, send the line under cursor."
  (interactive)
  (let ((text (string-trim (thing-at-point (if (use-region-p) 'region 'line) t))))
    (when text
      (aidermacs--send-command text))))

(defun aidermacs-send-region-by-line (start end)
  "Send the text between START and END, line by line.
Only sends non-empty lines after trimming whitespace."
  (interactive "r")
  (with-restriction start end
    (save-excursion
      (goto-char (point-min))
      (while (not (eobp))
        (let ((line (string-trim (thing-at-point 'line t))))
          (when (not (string-empty-p line))
            (aidermacs--send-command line)))
        (forward-line 1)))))

(defun aidermacs-send-block-or-region ()
  "Send the current active region text or current paragraph content.
When sending paragraph content, preserve cursor
position."
  (interactive)
  (let ((text (if (use-region-p)
                  (buffer-substring-no-properties
                   (region-beginning) (region-end))
                (save-excursion
                  (mark-paragraph)
                  (prog1
                      (buffer-substring-no-properties
                       (region-beginning) (region-end))
                    (deactivate-mark))))))
    (when text
      (aidermacs--send-command text))))

(defun aidermacs-open-prompt-file ()
  "Open aidermacs prompt file under project root.
If file doesn't exist, create it with command binding help and
sample prompt."
  (interactive)
  (let* ((root (aidermacs-project-root))
         (prompt-file (when root
                        (expand-file-name aidermacs-prompt-file-name root))))
    (if prompt-file
        (progn
          (find-file-other-window prompt-file)
          (unless (file-exists-p prompt-file)
            ;; Insert initial content for new file
            (insert "# aidermacs Prompt File - Command Reference:\n")
            (insert "# C-c C-n or C-<return>: Send current line or selected region line by line\n")
            (insert "# C-c C-c: Send current block or selected region as a whole\n")
            (insert "# C-c C-z: Switch to aidermacs buffer\n\n")
            (insert "* Sample task:\n\n")
            (insert "/ask what this repo is about?\n")
            (save-buffer)))
      (user-error "Could not determine prompt file"))))

;;;###autoload
(defvar aidermacs-minor-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-n") #'aidermacs-send-line-or-region)
    (define-key map (kbd "C-<return>") #'aidermacs-send-line-or-region)
    (define-key map (kbd "C-c C-c") #'aidermacs-send-block-or-region)
    (define-key map (kbd "C-c C-z") #'aidermacs-switch-to-buffer)
    map)
  "Keymap for `aidermacs-minor-mode'.")

;;;###autoload
(define-minor-mode aidermacs-minor-mode
  "Minor mode for interacting with aidermacs AI pair programming tool.

Provides these keybindings:
\\{aidermacs-minor-mode-map}"
  :lighter " aidermacs"
  :keymap aidermacs-minor-mode-map)

;; Auto-enable aidermacs-minor-mode for specific files
(defcustom aidermacs-auto-mode-files
  (list
   aidermacs-prompt-file-name    ; Default prompt file
   ".aider.chat.md"
   ".aider.chat.history.md"
   ".aider.input.history")
  "List of filenames that should automatically enable `aidermacs-minor-mode'.
These are exact filename matches (including the dot prefix)."
  :type '(repeat string))

(defun aidermacs--maybe-enable-minor-mode ()
  "Determines whether to enable `aidermacs-minor-mode'."
  (when (and buffer-file-name
             (member (file-name-nondirectory buffer-file-name)
                     aidermacs-auto-mode-files))
    (aidermacs-minor-mode 1)))

;;;###autoload
(defun aidermacs-setup-minor-mode ()
  "Set up automatic enabling of `aidermacs-minor-mode' for specific files.
This adds a hook to automatically enable the minor mode for files
matching patterns in `aidermacs-auto-mode-files'.
Only adds the hook if it's not already present.

The minor mode provides convenient keybindings for working with
prompt files and other Aider-related files:
\\<aidermacs-minor-mode-map>
\\[aidermacs-send-line-or-region] - Send current line/region line-by-line
\\[aidermacs-send-block-or-region] - Send block/region as whole
\\[aidermacs-switch-to-buffer] - Switch to Aidermacs buffer"
  (interactive)
  (unless (member #'aidermacs--maybe-enable-minor-mode find-file-hook)
    (add-hook 'find-file-hook #'aidermacs--maybe-enable-minor-mode)))

;;;###autoload
(defun aidermacs-switch-to-code-mode ()
  "Switch aider to code mode.
In code mode, aider will make changes to your code to satisfy
your requests."
  (interactive)
  (aidermacs--send-command "/chat-mode code")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'code))
  (message "Switched to code mode <default> - aider will make changes to your code"))

;;;###autoload
(defun aidermacs-switch-to-ask-mode ()
  "Switch aider to ask mode.
In ask mode, aider will answer questions about your code, but
never edit it."
  (interactive)
  (aidermacs--send-command "/chat-mode ask")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'ask))
  (message "Switched to ask mode - you can chat freely, aider will not edit your code"))

;;;###autoload
(defun aidermacs-switch-to-architect-mode ()
  "Switch aider to architect mode.
In architect mode, aider will first propose a solution, then ask
if you want it to turn that proposal into edits to your files."
  (interactive)
  (aidermacs--send-command "/chat-mode architect")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'architect))
  (message "Switched to architect mode - aider will propose solutions before making changes"))

;;;###autoload
(defun aidermacs-switch-to-help-mode ()
  "Switch aider to help mode.
In help mode, aider will answer questions about using aider,
configuring, troubleshooting, etc."
  (interactive)
  (aidermacs--send-command "/chat-mode help")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'help))
  (message "Switched to help mode - aider will answer questions about using aider"))

(defun aidermacs-refresh-repo-map ()
  "Force a refresh of the repository map.
This updates aider's understanding of the repository structure and files."
  (interactive)
  (aidermacs--send-command "/map-refresh")
  (message "Refreshing repository map..."))

(defun aidermacs-send-voice ()
  "send /voice command to aidermacs"
  (interactive)
  (aidermacs--send-command "/voice")
  (message "aidermacs awaiting speech"))

(defun aidermacs-web (url)
  "Fetch web content from URL using aider's web command.
This allows aider to access online documentation, references, or examples."
  (interactive "sEnter URL to fetch: ")
  (when (and url (not (string-empty-p url)))
    (aidermacs--send-command (format "/web %s" url))
    (message "Fetching content from %s..." url)))

;; Add a hook to clean up temp buffers when an aidermacs buffer is killed
(defun aidermacs--cleanup-on-buffer-kill ()
  "Clean up temporary buffers when an aidermacs buffer is killed."
  (when (aidermacs--is-aidermacs-buffer-p)
    (aidermacs--cleanup-temp-buffers)))

(defun aidermacs--setup-cleanup-hooks ()
  "Set up hooks to ensure proper cleanup of temporary buffers.
Only adds the hook if it's not already present."
  (unless (member #'aidermacs--cleanup-on-buffer-kill kill-buffer-hook)
    (add-hook 'kill-buffer-hook #'aidermacs--cleanup-on-buffer-kill)))

(provide 'aidermacs)
;;; aidermacs.el ends here

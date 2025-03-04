;;; aidermacs.el --- Aidermacs package for interactive conversation with aider -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 0.9.0
;; Package-Requires: ((emacs "28.1"))
;; Keywords: ai emacs agents llm aider ai-pair-programming, convenience, tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; Originally forked from: Kang Tu <tninja@gmail.com> Aider.el
;; SPDX-License-Identifier: Apache-2.0
;;
;; This file is not part of GNU Emacs.
;;
;;; Commentary:
;;
;; Aidermacs is an Emacs package that integrates with Aider, an AI
;; pair programming tool, to bring AI-powered code modification and
;; assistance to your Emacs workflow. It allows you to interact with
;; Aider through an Emacs buffer, providing commands for adding and
;; dropping files, asking questions, generating code, and more.
;;
;; Key features include:
;; - Integration with Aider for AI-assisted code modification.
;; - Support for multiple backends, including Comint and VTerm.
;; - Commands for managing files, asking questions, and generating code.
;; - Transient menus for easy access to commands.
;; - Minor mode for convenient keybindings in prompt files.
;;
;;; Code:

(require 'comint)
(require 'dired)
(require 'ediff)
(require 'project)
(require 'transient)
(require 'vc-git)
(require 'which-func)
(require 'ansi-color)
(require 'cl-lib)

(require 'aidermacs-backends)
(require 'aidermacs-models)

(defvar-local aidermacs--current-mode nil
  "Buffer-local variable to track the current aidermacs mode.
Possible values: 'code, 'ask, 'architect, 'help.")

(declare-function magit-show-commit "magit-diff" (rev &optional noselect module))

(defgroup aidermacs nil
  "Customization group for the aidermacs package."
  :prefix "aidermacs-"
  :group 'convenience)

(defcustom aidermacs-program "aider"
  "The name or path of the aidermacs program."
  :type 'string
  :group 'aidermacs)

(define-obsolete-variable-alias 'aidermacs-args 'aidermacs-extra-args "0.5.0"
  "Old name for `aidermacs-extra-args', please update your config.")

(defcustom aidermacs-config-file nil
  "Path to aider configuration file.
When set, Aidermacs will pass this to aider via --config flag,
ignoring other configuration settings except `aidermacs-extra-args'."
  :type '(choice (const :tag "None" nil)
                 (file :tag "Config file"))
  :group 'aidermacs)

(defcustom aidermacs-extra-args '()
  "Additional arguments to pass to the aidermacs command."
  :type '(repeat string)
  :group 'aidermacs)

(defcustom aidermacs-subtree-only nil
  "When non-nil, run aider with --subtree-only in the current directory.
This is useful for working in monorepos where you want to limit aider's scope."
  :type 'boolean
  :group 'aidermacs)

(defcustom aidermacs-auto-commits nil
  "When non-nil, enable auto-commits of LLM changes.
When nil, disable auto-commits requiring manual git commits."
  :type 'boolean
  :group 'aidermacs)

(defun aidermacs-project-root ()
  "Get the project root using project.el, VC, or fallback to file directory.
This function tries multiple methods to determine the project root."
  (or (when-let ((proj (project-current)))
          (project-root proj))
      (vc-git-root default-directory)
      (when buffer-file-name
        (file-name-directory buffer-file-name))
      default-directory))

(defcustom aidermacs-prompt-file-name ".aider.prompt.org"
  "File name that will automatically enable `aidermacs-minor-mode' when opened.
This is the file name without path."
  :type 'string
  :group 'aidermacs)

(defvar-local aidermacs-read-string-history nil
  "History list for aidermacs read string inputs.")

(defvar-local aidermacs--pre-edit-files nil
  "Alist of (filename . temp-filename) pairs storing file state before Aider edits.")

(defvar-local aidermacs--pre-edit-buffers nil
  "Alist of (filename . temp-buffer) pairs for active ediff sessions.")


;;;###autoload
(defun aidermacs-plain-read-string (prompt &optional initial-input)
  "Read a string from the user with PROMPT and optional INITIAL-INPUT.
PROMPT is the text to display.  INITIAL-INPUT is the default value."
  (read-string prompt initial-input 'aidermacs-read-string-history))

;;;###autoload
(defalias 'aidermacs-read-string #'aidermacs-plain-read-string)

(eval-and-compile
  ;; Ensure the alias is always available in both compiled and interpreted modes.
  (defalias 'aidermacs-read-string #'aidermacs-plain-read-string))

;; Main transient menu
(transient-define-prefix aidermacs-transient-menu ()
  "AI Pair Programming Interface."
  ["Aidermacs: AI Pair Programming"
   ["Core"
    ("a" "Start/Open Session" aidermacs-run)
    ("." "Start in Current Dir" aidermacs-run-in-current-dir)
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
    ("h" "Session History" aidermacs-show-output-history)
    ("o" "Change Main Model" aidermacs-change-model)
    ("O" "Clear Selection Cache" aidermacs-clear-model-cache)
    ("?" "Aider Help" aidermacs-help)]]
  ["File Actions"
   ["Add Files (C-u: read-only)"
    ("f" "Add File" aidermacs-add-file)
    ("F" "Add Current File" aidermacs-add-current-file)
    ("d" "Add From Directory (same type)" aidermacs-add-same-type-files-under-dir)
    ("w" "Add From Window" aidermacs-add-files-in-current-window)
    ("m" "Add From Dired (marked)" aidermacs-batch-add-dired-marked-files)]
   ["Drop Files"
    ("j" "Drop File" aidermacs-drop-file)
    ("J" "Drop Current File" aidermacs-drop-current-file)
    ("k" "Drop All Files" aidermacs-drop-all-files)]
   ["Others"
    ("S" "Create Session Scratchpad" aidermacs-create-session-scratchpad)
    ("G" "Add File to Session" aidermacs-add-file-to-session)
    ("A" "List Added Files" aidermacs-list-added-files)]]
  ["Code Actions"
   ["Architect"
    ("c" "General Architect" aidermacs-general-architect)
    ("r" "Architect This Code" aidermacs-architect-this-code)
    ("i" "Implement TODO" aidermacs-implement-todo)]
   ["Question"
    ("q" "General Question" aidermacs-general-question)
    ("e" "Question This Code" aidermacs-question-this-code)
    ("p" "Question This Symbol" aidermacs-question-this-symbol)
    ("g" "Accept Changes" aidermacs-accept-change)]
   ["Others"
    ("RET" "Code Change Now" aidermacs-direct-change)
    ("t" "Write Test" aidermacs-write-unit-test)
    ("T" "Fix Test" aidermacs-fix-failing-test-under-cursor)
    ("!" "Debug Exception" aidermacs-debug-exception)]])

(defun aidermacs-select-buffer-name ()
  "Select an existing aidermacs session buffer.
If there is only one aidermacs buffer, return its name.
If there are multiple, prompt to select one interactively.
Returns nil if no aidermacs buffers exist.
This is used when you want to target an existing session."
  (let* ((buffers (cl-remove-if-not
                   (lambda (buf)
                     (string-match-p "^\\*aidermacs:" (buffer-name buf)))
                   (buffer-list)))
         (buffer-names (mapcar #'buffer-name buffers)))
    (cond
     ((null buffers) nil)
     ((= (length buffers) 1) (car buffer-names))
     (t (completing-read "Select aidermacs session: " buffer-names nil t)))))

(defun aidermacs-get-buffer-name (&optional use-existing)
  "Generate the aidermacs buffer name based on project root or current directory.
If USE-EXISTING is non-nil, use an existing buffer instead of creating new."
  (if use-existing
      (aidermacs-select-buffer-name)
  (let* ((root (aidermacs-project-root))
         (current-dir (file-truename default-directory))
         ;; Get all existing aidermacs buffers
         (aidermacs-buffers
          (cl-remove-if-not
           (lambda (buf)
             (string-match-p "^\\*aidermacs:" (buffer-name buf)))
           (buffer-list)))
         ;; Extract directory paths and subtree status from buffer names
         (buffer-dirs
          (mapcar
           (lambda (buf)
             (when (string-match "^\\*aidermacs:\\(.*?\\)\\*$"
                                 (buffer-name buf))
               (cons (match-string 1 (buffer-name buf))
                     (match-string 2 (buffer-name buf)))))
           aidermacs-buffers))
         ;; Find closest parent directory that has an aidermacs session
         (closest-parent
          (car
           (car
            (sort
             (cl-remove-if-not
              (lambda (dir-info)
                (and (car dir-info)
                     (string-prefix-p (car dir-info) current-dir)
                     (file-exists-p (car dir-info))))
              buffer-dirs)
             (lambda (a b)
               ;; Sort by path length (deeper paths first)
               (> (length (car a)) (length (car b))))))))
         (display-root (cond
                        ;; Use current directory for new subtree session
                        (aidermacs-subtree-only current-dir)
                        ;; Use closest parent if it exists
                        (closest-parent closest-parent)
                        ;; Fall back to project root for new non-subtree session
                        (t root))))
    (format "*aidermacs:%s*"
            (file-truename display-root)))))

;;;###autoload
(defun aidermacs-run ()
  "Run aidermacs process using the selected backend.
This function sets up the appropriate arguments and launches the process."
  (interactive)
  (let* ((buffer-name (aidermacs-get-buffer-name))
         ;; Process extra args: split each string on whitespace.
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
         (backend-args
          (if has-config-arg
              ;; Only need to add aidermacs-config-file manually
              (when aidermacs-config-file
                (list "--config" aidermacs-config-file))
            (append
             (if aidermacs-use-architect-mode
                 (list "--architect"
                       "--model" aidermacs-architect-model
                       "--editor-model" aidermacs-editor-model)
               (unless has-model-arg
                 (list "--model" aidermacs-default-model)))
             (when (not aidermacs-auto-commits)
               '("--no-auto-commits"))
             (when aidermacs-subtree-only
               '("--subtree-only")))))
         (final-args (append backend-args flat-extra-args)))
    (if (get-buffer buffer-name)
        (aidermacs-switch-to-buffer buffer-name)
      (aidermacs-run-backend aidermacs-program final-args buffer-name)
      (with-current-buffer (get-buffer buffer-name)
        ;; Set initial mode based on startup configuration
        (setq-local aidermacs--current-mode (if aidermacs-use-architect-mode 'architect 'code)))
      (aidermacs-switch-to-buffer buffer-name))))

;;;###autoload
(defun aidermacs-run-in-current-dir ()
  "Run aidermacs in the current directory with --subtree-only flag.
This is useful for working in monorepos where you want to limit aider's scope."
  (interactive)
  (let ((aidermacs-subtree-only t)
        (default-directory (file-truename default-directory)))
    (aidermacs-run)))

(defun aidermacs--capture-file-state (filename)
  "Store the current state of FILENAME in a temporary file."
  (when (and filename (file-exists-p filename))
    (let ((temp-file (make-temp-file
                      (concat "aidermacs-"
                              (file-name-nondirectory filename) "-"))))
      (condition-case err
          (progn
            (copy-file filename temp-file t)
            (cons filename temp-file))
        (error
         (message "Error capturing file state for %s: %s"
                  filename (error-message-string err))
         (when (file-exists-p temp-file)
           (delete-file temp-file))
         nil)))))

(defun aidermacs--prepare-for-code-edit ()
  "Prepare for Aider code edits by capturing current file states."
  (let ((files (when (boundp 'aidermacs--tracked-files)
                 aidermacs--tracked-files)))
    (when files
      ;; Initialize the tracking variables if they don't exist
      (unless (boundp 'aidermacs--pre-edit-files)
        (setq-local aidermacs--pre-edit-files nil))
      (unless (boundp 'aidermacs--pre-edit-buffers)
        (setq-local aidermacs--pre-edit-buffers nil))

      (setq aidermacs--pre-edit-files
            (mapcar (lambda (file)
                      (let ((clean-file (replace-regexp-in-string " (read-only)$" "" file)))
                        (aidermacs--capture-file-state
                         (expand-file-name clean-file (aidermacs-project-root)))))
                    files)))))

(defun aidermacs-cleanup-all-ediff-sessions ()
  "Quit all active aidermacs ediff sessions and clean up all resources.
This function ensures a complete cleanup of all ediff-related resources."
  (interactive)
  ;; Find and quit all active aidermacs ediff sessions
  (dolist (buf (buffer-list))
    (when (string-match-p "\\*ediff-aidermacs:" (buffer-name buf))
      (with-current-buffer buf
        ;; This will trigger the normal ediff cleanup process
        (when (fboundp 'ediff-quit)
          (ediff-quit)))))

  ;; Force cleanup of any remaining resources
  (aidermacs--cleanup-stale-pre-edit-buffers))

(defun aidermacs--cleanup-stale-pre-edit-buffers ()
  "Clean up any stale pre-edit buffers that weren't properly disposed of.
This function should only handle truly orphaned resources that weren't
cleaned up by the normal ediff quit process."
  ;; Clean up any orphaned pre-edit buffers
  (when (boundp 'aidermacs--pre-edit-buffers)
    (dolist (pair aidermacs--pre-edit-buffers)
      (let ((file (car pair))
            (buffer (cdr pair)))
        (when (and buffer (buffer-live-p buffer))
          (message "Cleaning up orphaned pre-edit buffer for %s" file)
          (kill-buffer buffer)))))

  ;; Clean up any orphaned temp files
  (when (boundp 'aidermacs--pre-edit-files)
    (dolist (pair aidermacs--pre-edit-files)
      (let ((file (car pair))
            (temp-file (cdr pair)))
        (when (and temp-file (stringp temp-file) (file-exists-p temp-file))
          (message "Cleaning up orphaned temp file for %s" file)
          (delete-file temp-file))))))

(defun aidermacs--setup-ediff-cleanup-hooks ()
  "Set up hooks to ensure proper cleanup of temporary buffers after ediff."
  (add-hook 'ediff-quit-hook #'aidermacs--cleanup-stale-pre-edit-buffers)
  ;; Also add to kill-emacs-hook to ensure cleanup on Emacs exit
  (add-hook 'kill-emacs-hook #'aidermacs-cleanup-all-ediff-sessions))


(defun aidermacs--detect-edited-files ()
  "Parse current output to find files edited by Aider."
  (let ((edited-files nil)
        (output aidermacs--current-output))
    (with-temp-buffer
      (insert output)
      (goto-char (point-min))
      (while (re-search-forward "Applied edit to \\(.+\\)" nil t)
        (push (match-string 1) edited-files)))
    (nreverse edited-files)))

(defun aidermacs--create-pre-edit-buffer (filename temp-file)
  "Create a buffer for FILENAME using content from TEMP-FILE for ediff."
  (condition-case err
      (let ((buffer (generate-new-buffer (format "*aidermacs-pre-edit:%s*"
                                                (file-name-nondirectory filename)))))
        (with-current-buffer buffer
          (condition-case err2
              (progn
                (insert-file-contents temp-file)
                (set-buffer-modified-p nil)
                ;; Use same major mode as the original file would have
                (let ((buffer-file-name filename))
                  (set-auto-mode))
                (setq buffer-read-only t))
            (error
             (message "Error setting up pre-edit buffer: %s" (error-message-string err2))
             (kill-buffer buffer)
             nil)))
        buffer)
    (error
     (message "Failed to create pre-edit buffer: %s" (error-message-string err))
     nil)))

(defun aidermacs--show-ediff-for-edited-files (edited-files)
  "Show ediff for each file in EDITED-FILES."
  (when edited-files
    ;; Display a message about which files were changed
    (message "Modified %d file(s): %s"
             (length edited-files)
             (mapconcat #'identity edited-files ", "))

    (let ((file (car edited-files))
          (remaining (cdr edited-files)))

      ;; Find the pre-edit temp file for this file
      (let* ((full-path (expand-file-name file (aidermacs-project-root)))
             (pre-edit-pair (when (boundp 'aidermacs--pre-edit-files)
                              (assoc full-path aidermacs--pre-edit-files)))
             (temp-file (and pre-edit-pair (cdr pre-edit-pair))))

        (if (and temp-file
                 (stringp temp-file)
                 (file-exists-p temp-file))
            (progn
              ;; Create buffer from temp file only when needed
              (let ((pre-edit-buffer (aidermacs--create-pre-edit-buffer full-path temp-file)))
                ;; Store buffer for cleanup
                (unless (boundp 'aidermacs--pre-edit-buffers)
                  (setq-local aidermacs--pre-edit-buffers nil))
                (push (cons full-path pre-edit-buffer) aidermacs--pre-edit-buffers)

                ;; Start ediff session
                (ediff-buffers
                 pre-edit-buffer
                 (or (get-file-buffer full-path)
                     (find-file-noselect full-path))
                 ;; Setup callback for when this ediff session ends
                 (lambda ()
                   ;; Clean up resources immediately and directly
                   ;; 1. Kill the pre-edit buffer immediately
                   (when (and pre-edit-buffer (buffer-live-p pre-edit-buffer))
                     (kill-buffer pre-edit-buffer))
                   ;; 2. Remove this file from the buffer tracking list
                   (when (boundp 'aidermacs--pre-edit-buffers)
                     (setq aidermacs--pre-edit-buffers
                           (assoc-delete-all full-path aidermacs--pre-edit-buffers)))
                   ;; 3. Delete the temp file immediately
                   (when (and temp-file (stringp temp-file) (file-exists-p temp-file))
                     (delete-file temp-file))
                   ;; 4. Remove this file from the temp file tracking list
                   (when (boundp 'aidermacs--pre-edit-files)
                     (setq aidermacs--pre-edit-files
                           (assoc-delete-all full-path aidermacs--pre-edit-files)))
                   ;; Process remaining files
                   (when remaining
                     (aidermacs--show-ediff-for-edited-files remaining)))))

              ;; Customize ediff session
              (when (boundp 'ediff-buffer)
                (with-current-buffer ediff-buffer
                  (rename-buffer (format "*ediff-aidermacs: %s*" file)))))

          ;; If no pre-edit temp file found, continue with remaining files
          (when remaining
            (aidermacs--show-ediff-for-edited-files remaining)))))))

(defun aidermacs--send-command-with-edit-tracking (buffer command)
  "Send COMMAND to aidermacs with edit tracking to the BUFFER."
  ;; Prepare for edits
  (aidermacs--prepare-for-code-edit)
  ;; Send command and track output
  (aidermacs--send-command-backend
   buffer command nil
   (lambda ()
     ;; Check if this was a code edit command and show ediff if needed
     (when (aidermacs--is-edit-command-p command buffer)
       (let ((edited-files (aidermacs--detect-edited-files)))
         (when edited-files
           (aidermacs--show-ediff-for-edited-files edited-files)))))))

(defun aidermacs--is-edit-command-p (command buffer)
  "Determine if COMMAND might result in code edits based on current mode in BUFFER.
Returns non-nil if the command is a direct edit command or if we're in a mode
that allows edits and the command isn't explicitly a non-edit command."
  (with-current-buffer buffer
    (or
     ;; Explicit edit commands always trigger edit tracking
     (string-match-p "^\\(/code\\|/architect\\)" command)

     ;; In code or architect mode, any message that's not a command could edit code
     (and (memq aidermacs--current-mode '(code architect))
          (not (string-prefix-p "/" command))))))

(defun aidermacs--send-command (command &optional no-switch-to-buffer use-existing redirect callback)
  "Send command to the corresponding aidermacs process.
COMMAND is the text to send.
If NO-SWITCH-TO-BUFFER is non-nil, don't switch to the aidermacs buffer.
If USE-EXISTING is non-nil, use an existing buffer instead of creating new.
If REDIRECT is non-nil it redirects the output (hidden) for comint backend.
If CALLBACK is non-nil it will be called after the command finishes."
  (let* ((buffer-name (aidermacs-get-buffer-name use-existing))
         (buffer (or (get-buffer buffer-name)
                     (progn (aidermacs-run)
                            (get-buffer buffer-name))))
         (processed-command (aidermacs--process-message-if-multi-line command)))

    ;; Reset current output before sending new command
    (with-current-buffer buffer
      (setq-local aidermacs--current-output nil))

    (if (aidermacs--is-edit-command-p command buffer)
        (aidermacs--send-command-with-edit-tracking buffer processed-command)
      (aidermacs--send-command-backend buffer processed-command redirect callback))
    (when (and (not no-switch-to-buffer) (not (string= (buffer-name) buffer-name)))
      (aidermacs-switch-to-buffer buffer-name))))

;;;###autoload
(defun aidermacs-switch-to-buffer (&optional buffer-name)
  "Switch to the aidermacs buffer.
If BUFFER-NAME is provided, switch to that buffer.
If the buffer is already visible in a window, switch to that window.
If the current buffer is already the aidermacs buffer, do nothing."
  (interactive)
  (let* ((target-buffer-name (or buffer-name (aidermacs-get-buffer-name)))
         (buffer (get-buffer target-buffer-name)))
    (cond
     ((string= (buffer-name) target-buffer-name) t)
     ((and buffer (get-buffer-window buffer))
      (select-window (get-buffer-window buffer)))  ;; Switch to existing window
     (buffer
      (pop-to-buffer buffer))
     (t
      (message "Buffer '%s' does not exist." target-buffer-name)))))

;;;###autoload
(defun aidermacs-clear-chat-history ()
  "Send the command \"/clear\" to the aidermacs buffer."
  (interactive)
  (setq aidermacs--tracked-files nil)
  (aidermacs--send-command "/clear"))

;;;###autoload
(defun aidermacs-reset ()
  "Send the command \"/reset\" to the aidermacs buffer."
  (interactive)
  (setq aidermacs--tracked-files nil)
  (aidermacs--send-command "/reset"))

;;;###autoload
(defun aidermacs-exit ()
  "Send the command \"/exit\" to the aidermacs buffer."
  (interactive)
  (aidermacs--send-command "/exit" t))


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

;;;###autoload
(defun aidermacs-act-on-current-file (command-prefix)
  "Send a command with the current file path to the aidermacs buffer.
COMMAND-PREFIX is the command to prepend to the file path.
The full command will be \"COMMAND-PREFIX <current buffer file full path>\"."
  ;; Ensure the current buffer is associated with a file
  (if (not buffer-file-name)
      (message "Current buffer is not associated with a file.")
    (let* ((file-path (buffer-file-name))
           ;; Use buffer-file-name directly
           (formatted-path (if (string-match-p " " file-path)
                               (format "\"%s\"" file-path)
                             file-path))
           (command (format "%s %s" command-prefix formatted-path)))
      ;; Use the shared helper function to send the command
      (aidermacs--send-command command))))

;;;###autoload
(defun aidermacs-drop-current-file ()
  "Drop the current file from aidermacs session."
  (interactive)
  (aidermacs-act-on-current-file "/drop"))

;;;###autoload
(defun aidermacs-general-command ()
  "Prompt the user to input COMMAND and send it to the aidemracs."
  (interactive)
  (let ((command (aidermacs-read-string "Enter general aider command: ")))
    ;; Use the shared helper function to send the command
    (aidermacs--send-command command)))

;;;###autoload
(defun aidermacs-direct-change ()
  "Prompt the user for an input and send it to aidemracs prefixed with \"/code \"."
  (interactive)
  (when-let ((command (aidermacs--form-prompt "/code" nil t "empty to change to code mode")))
    (aidermacs--send-command command)))

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
      (let ((files '())
            (base (or (vc-git-root default-directory)
                      default-directory)))
        ;; Parse read-only files section
        (when (search-forward "Read-only files:" nil t)
          (forward-line 1)
          (while (and (not (eobp))
                      (string-match-p "^[[:space:]]" (thing-at-point 'line t)))
            (let* ((line (string-trim (thing-at-point 'line t)))
                   (file (car (split-string line))))
              (when (and file (file-exists-p (expand-file-name file base)))
                (push (concat file " (read-only)") files)))
            (forward-line 1)))

        ;; Parse files in chat section
        (when (search-forward "Files in chat:" nil t)
          (forward-line 1)
          (while (and (not (eobp))
                      (string-match-p "^[[:space:]]" (thing-at-point 'line t)))
            (let* ((line (string-trim (thing-at-point 'line t)))
                   (file (car (split-string line))))
              (when (and file (file-exists-p (expand-file-name file base)))
                (push file files)))
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

;;;###autoload
(defun aidermacs-list-added-files ()
  "List all files currently added to the chat session.
Sends the \"/ls\" command and returns the list of files via callback."
  (interactive)
  (aidermacs--get-files-in-session
   (lambda (files)
     (message "%s" (prin1-to-string files))
     files)))

;;;###autoload
(defun aidermacs-drop-file ()
  "Drop a file from the chat session by selecting from currently added files."
  (interactive)
  (aidermacs--get-files-in-session
   (lambda (files)
     (if-let* ((file (completing-read "Select file to drop: " files nil t))
               (clean-file (replace-regexp-in-string " (read-only)$" "" file)))
         (aidermacs--send-command (format "/drop ./%s" clean-file))
       (message "No files available to drop")))))


;;;###autoload
(defun aidermacs-drop-all-files ()
  "Drop all files from the current chat session."
  (interactive)
  (aidermacs--send-command "/drop"))


;;;###autoload
(defun aidermacs-show-output-history ()
  "Display the AI output history in a new buffer."
  (interactive)
  (let ((buf (get-buffer-create "*aidermacs-history*"))
        (history aidermacs--output-history))
    (with-current-buffer buf
      (org-mode)
      (setq buffer-read-only nil)
      (erase-buffer)
      (display-line-numbers-mode 1)
      (dolist (entry history)
        (let ((timestamp (format-time-string "%F %T" (car entry)))
              (output (cdr entry)))
          (insert (format "* %s\n#+BEGIN_SRC\n%s\n#+END_SRC\n" timestamp output))))
      (goto-char (point-min))
      (setq buffer-read-only t)
      (local-set-key (kbd "q") 'kill-this-buffer)
      (switch-to-buffer-other-window buf))))

;;;###autoload
(defun aidermacs-get-last-output ()
  "Get the most recent output from aidermacs."
  (interactive)
  (when (stringp aidermacs--current-output)
    (message "%s" aidermacs--current-output)
    (kill-new aidermacs--current-output)
    aidermacs--current-output))


;;;###autoload
(defun aidermacs-question-this-code ()
  "Ask a question about the code at point or region.
If a region is active, include the region text in the question.
If cursor is inside a function, include the function name as context.
If called from the aidermacs buffer, use general question instead."
  (interactive)
  ;; Dispatch to general question if in aidermacs buffer
  (if (string= (buffer-name) (aidermacs-get-buffer-name))
      (call-interactively #'aidermacs-general-question)
    (when-let ((command (aidermacs--form-prompt "/ask" "Question")))
      (aidermacs-add-current-file)
      (aidermacs--send-command command))))

;;;###autoload
(defun aidermacs-general-question ()
  "Prompt the user for a general question without code context."
  (interactive)
  (when-let ((command (aidermacs--form-prompt "/ask" nil t "empty to change to ask mode")))
    (aidermacs--send-command command)))

;;;###autoload
(defun aidermacs-help ()
  "Prompt the user for an input prefixed with \"/help \"."
  (interactive)
  (when-let ((command (aidermacs--form-prompt "/help" nil t "empty for general /help")))
    (aidermacs--send-command command)))

;;;###autoload
(defun aidermacs-general-architect ()
  "Prompt the user for an input prefixed with \"/architect \"."
  (interactive)
  (when-let ((command (aidermacs--form-prompt "/architect" nil t "empty to change to architect mode")))
    (aidermacs--send-command command)))

;;;###autoload
(defun aidermacs-debug-exception ()
  "Prompt the user for an input and send it to aidemracs prefixed with \"/debug \"."
  (interactive)
  (when-let ((command (aidermacs--form-prompt "/ask" "Debug exception")))
    (aidermacs--send-command command)))

;;;###autoload
(defun aidermacs-accept-change ()
  "Send the command \"go ahead\" to the aidemracs."
  (interactive)
  (aidermacs--send-command "/code go ahead"))


;;;###autoload
(defun aidermacs-magit-show-last-commit ()
  "Show the last commit message using Magit.
If Magit is not installed, report that it is required."
  (interactive)
  (if (require 'magit nil 'noerror)
      (magit-show-commit "HEAD")
    (message "Magit is required to show the last commit.")))

;;;###autoload
(defun aidermacs-undo-last-commit ()
  "Undo the last change made by aidermacs."
  (interactive)
  (aidermacs--send-command "/undo"))

(defun aidermacs--form-prompt (command prompt-prefix &optional ignore-context guide)
  "Get command based on context with COMMAND and PROMPT-PREFIX.
COMMAND is the text to prepend.  PROMPT-PREFIX is the text to add after COMMAND.
If IGNORE-CONTEXT is non-nil, skip function and region context.
If region is active, use that region's text.
If point is in a function, use function name.
GUIDE is displayed in the prompt but not included in the final command."
  (let* ((on-function (unless ignore-context (which-function)))
         (region-text (when (and (use-region-p) (not ignore-context))
                        (buffer-substring-no-properties (region-beginning) (region-end))))
         (context (concat (when on-function
                            (format " in function `%s`" on-function))
                          (when region-text
                            (format " on code block:\n```\n%s\n```\n" region-text))))
         (prompt (concat command " " prompt-prefix context
                        (when guide (format "(%s)" guide)) ": "))
         (user-command (aidermacs-read-string prompt)))
    (concat command (unless (string= user-command "")
                      (concat " " prompt-prefix context ": " user-command)))))

(defun aidermacs-architect-this-code ()
  "Architect code at point or region.
If region is active, inspect that region.
If point is in a function, inspect that function."
  (interactive)
  ;; Dispatch to general architect if in aidermacs buffer
  (if (string= (buffer-name) (aidermacs-get-buffer-name))
      (call-interactively #'aidermacs-general-architect)
    (when-let ((command (aidermacs--form-prompt "/architect" "Architect")))
      (aidermacs-add-current-file)
      (aidermacs--send-command command))))

;;;###autoload
(defun aidermacs-question-this-symbol ()
  "Ask aidermacs to explain symbol under point."
  (interactive)
  (let* ((symbol (thing-at-point 'symbol))
         (line (buffer-substring-no-properties
                (line-beginning-position)
                (line-end-position)))
         (prompt (format "/ask Please explain what '%s' means in the context of this code line: %s"
                         symbol line)))
    (if symbol
        (progn
          (aidermacs-add-current-file)
          (aidermacs--send-command prompt))
      (error "No symbol under point!"))))

(defun aidermacs-send-command-with-prefix (prefix command)
  "Send COMMAND to the aidermacs buffer with PREFIX.
PREFIX is the text to prepend.  COMMAND is the text to send."
  (aidermacs-add-current-file)
  (aidermacs--send-command (concat prefix command)))

(defun aidermacs--add-files-helper (files read-only &optional message)
  "Helper function to add files with read-only flag.
FILES is a list of file paths to add.  READ-ONLY determines if files are added
as read-only.  Optional MESSAGE can override the default success message."
  (let* ((cmd (if read-only "/read-only" "/add"))
         (files (delq nil files)))
    (if files
        (progn
          (aidermacs--send-command
           (format "%s %s" cmd
                   (mapconcat #'identity files " ")))
          (message (or message
                       (format "Added %d files as %s"
                               (length files)
                               (if read-only "read-only" "editable")))))
      (message "No files to add."))))

;;;###autoload
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


;;;###autoload
(defun aidermacs-add-file (&optional read-only)
  "Add file(s) to aidermacs interactively.
With prefix argument `C-u', add as READ-ONLY.
If current buffer is visiting a file, its name is used as initial input.
Multiple files can be selected by calling the command multiple times."
  (interactive "P")
  (let* ((initial (when buffer-file-name
                    (file-name-nondirectory buffer-file-name)))
         (file (expand-file-name
                (read-file-name "Select file to add: "
                                nil nil t initial))))
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

;;;###autoload
(defun aidermacs-add-files-in-current-window (&optional read-only)
  "Add window files with READ-ONLY flag.
With prefix argument `C-u', add as read-only."
  (interactive "P")
  (let* ((files (mapcar (lambda (buffer)
                          (with-current-buffer buffer
                            (when buffer-file-name
                              (expand-file-name buffer-file-name))))
                        (mapcar #'window-buffer (window-list))))
         (filtered-files (delq nil files)))
    (aidermacs--add-files-helper filtered-files read-only)))

;;;###autoload
(defun aidermacs-batch-add-dired-marked-files (&optional read-only)
  "Add Dired files with READ-ONLY flag.
With prefix argument `C-u', add as read-only."
  (interactive "P")
  (aidermacs--add-files-helper (dired-get-marked-files) read-only))

;;;###autoload
(defun aidermacs-add-same-type-files-under-dir (&optional read-only)
  "Add all files with same suffix as current file under current directory.
If there are more than 40 files, refuse to add and show warning message.
With prefix argument `C-u', add as READ-ONLY."
  (interactive "P")
  (if (not buffer-file-name)
      (message "Current buffer is not visiting a file")
    (let* ((current-suffix (file-name-extension buffer-file-name))
           (dir (file-name-directory buffer-file-name))
           (max-files 40)
           (files (directory-files dir t (concat "\\." current-suffix "$") t)))
      (if (> (length files) max-files)
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
      (message "Current buffer is not visiting a file.")
    (let ((is-test-file (string-match-p "test" (file-name-nondirectory buffer-file-name)))
          (function-name (which-function)))
      (cond
       ;; Test file case
       (is-test-file
        (if function-name
            (if (string-match-p "test" function-name)
                (let* ((initial-input
                        (format "Please implement test function '%s'. Follow standard unit testing practices and make it a meaningful test. Do not use Mock if possible."
                                function-name))
                       (command (aidermacs--form-prompt "/architect" initial-input)))
                  (aidermacs-add-current-file)
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
          (aidermacs-add-current-file)
          (aidermacs--send-command command)))))))

;;;###autoload
(defun aidermacs-fix-failing-test-under-cursor ()
  "Report the current test failure to aidermacs and ask it to fix the code.
This function assumes the cursor is on or inside a test function."
  (interactive)
  (if-let ((test-function-name (which-function)))
      (let* ((initial-input (format "The test '%s' is failing. Please analyze and fix the code to make the test pass. Don't break any other test"
                                    test-function-name))
             (command (aidermacs--form-prompt "/architect" initial-input)))
        (aidermacs-add-current-file)
        (aidermacs--send-command command))
    (message "No test function found at cursor position.")))

(defun aidermacs-create-session-scratchpad ()
  "Create a new temporary file for adding content to the aider session.
The file will be created in the system's temp directory with a timestamped name.
Use this to add functions, code snippets, or other content to the session."
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
    (aidermacs--send-command (format "/read %s" filename))
    (find-file-other-window filename)
    (message "Created and added scratchpad to session: %s" filename)))

;;;###autoload
(defun aidermacs-add-file-to-session ()
  "Interactively add a file to an existing aidermacs session using /read.
This allows you to add the file's content to a specific session."
  (interactive)
  (let* ((initial (when buffer-file-name
                    (file-name-nondirectory buffer-file-name)))
         (file (expand-file-name
                (read-file-name "Select file to add to existing session: "
                                nil nil t initial))))
    (if (not (file-exists-p file))
        (message "File does not exist: %s" file)
      (aidermacs--send-command (format "/read %s" file) nil t))))

(defun aidermacs--is-comment-line (line)
  "Check if LINE is a comment line based on current buffer's comment syntax.
Returns non-nil if LINE starts with one or more comment characters,
ignoring leading whitespace."
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
      (when-let ((command (aidermacs--form-prompt
                           "/architect"
                           (concat "Please implement the TODO items."
                                   (when is-comment
                                     (format " on this comment: `%s`." current-line))
                                   " Keep existing code structure"))))
        (aidermacs-add-current-file)
        (aidermacs--send-command command)))))

;;;###autoload
(defun aidermacs-send-line-or-region ()
  "Send text to the aidermacs buffer.
If region is active, send the selected region.
Otherwise, send the line under cursor."
  (interactive)
  (let ((text (if (use-region-p)
                  (buffer-substring-no-properties (region-beginning) (region-end))
                (string-trim (thing-at-point 'line t)))))
    (when text
      (aidermacs--send-command text))))

;;;###autoload
(defun aidermacs-send-region-by-line ()
  "Send the text of the current selected region, split into lines."
  (interactive)
  (if (use-region-p)
      (let* ((text (buffer-substring-no-properties (region-beginning) (region-end)))
             (lines (split-string text "\n" t)))
        (mapc (lambda (line)
                (let ((trimmed (string-trim line)))
                  (when (not (string-empty-p trimmed))
                    (aidermacs--send-command trimmed))))
              lines))
    (message "No region selected.")))

;;;###autoload
(defun aidermacs-send-block-or-region ()
  "Send the current active region text or current paragraph content.
When sending paragraph content, preserve cursor position."
  (interactive)
  (let ((text (if (use-region-p)
                  (buffer-substring-no-properties (region-beginning) (region-end))
                (save-excursion
                  (mark-paragraph)
                  (prog1
                      (buffer-substring-no-properties (region-beginning) (region-end))
                    (deactivate-mark))))))
    (when text
      (aidermacs--send-command text))))

;;;###autoload
(defun aidermacs-open-prompt-file ()
  "Open aidermacs prompt file under git repo root.
If file doesn't exist, create it with command binding help and sample prompt."
  (interactive)
  (let* ((git-root (vc-git-root default-directory))
         (prompt-file (when git-root
                        (expand-file-name aidermacs-prompt-file-name git-root))))
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
      (message "Not in a git repository"))))

;;;###autoload
(defvar aidermacs-minor-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-n") 'aidermacs-send-line-or-region)
    (define-key map (kbd "C-<return>") 'aidermacs-send-line-or-region)
    (define-key map (kbd "C-c C-c") 'aidermacs-send-block-or-region)
    (define-key map (kbd "C-c C-z") 'aidermacs-switch-to-buffer)
    map)
  "Keymap for `aidermacs-minor-mode'.")

;;;###autoload
(define-minor-mode aidermacs-minor-mode
  "Minor mode for interacting with aidermacs AI pair programming tool.

Provides these keybindings:
\\{aidermacs-minor-mode-map}"
  :lighter " aidermacs"
  :keymap aidermacs-minor-mode-map
  :override t
  :group 'aidermacs)

;; Auto-enable aidermacs-minor-mode for specific files
(defcustom aidermacs-auto-mode-files
  (list
   aidermacs-prompt-file-name    ; Default prompt file
   ".aider.chat.md"
   ".aider.chat.history.md"
   ".aider.input.history")
  "List of filenames that should automatically enable `aidermacs-minor-mode'.
These are exact filename matches (including the dot prefix)."
  :type '(repeat string)
  :group 'aidermacs)

(defun aidermacs--maybe-enable-minor-mode ()
  "Determines whether to enable `aidermacs-minor-mode'."
  (when (and buffer-file-name
             (when buffer-file-name
               (let ((base-name (file-name-nondirectory buffer-file-name)))
                 (member base-name aidermacs-auto-mode-files))))
    (aidermacs-minor-mode 1)))

;;;###autoload
(defun aidermacs-setup-minor-mode ()
  "Set up automatic enabling of `aidermacs-minor-mode' for specific files.
This adds a hook to automatically enable the minor mode for files
matching patterns in `aidermacs-auto-mode-files'.

The minor mode provides convenient keybindings for working with
prompt files and other Aider-related files:
\\<aidermacs-minor-mode-map>
\\[aidermacs-send-line-or-region] - Send current line/region line-by-line
\\[aidermacs-send-block-or-region] - Send block/region as whole
\\[aidermacs-switch-to-buffer] - Switch to Aidermacs buffer"
  (interactive)
  (add-hook 'find-file-hook #'aidermacs--maybe-enable-minor-mode))

;;;###autoload
(defun aidermacs-switch-to-code-mode ()
  "Switch aider to code mode.
In code mode, aider will make changes to your code to satisfy your requests."
  (interactive)
  (aidermacs--send-command "/chat-mode code")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'code))
  (message "Switched to code mode <default> - aider will make changes to your code"))

;;;###autoload
(defun aidermacs-switch-to-ask-mode ()
  "Switch aider to ask mode.
In ask mode, aider will answer questions about your code, but never edit it."
  (interactive)
  (aidermacs--send-command "/chat-mode ask")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'ask))
  (message "Switched to ask mode - you can chat freely, aider will not edit your code"))

;;;###autoload
(defun aidermacs-switch-to-architect-mode ()
  "Switch aider to architect mode.
In architect mode, aider will first propose a solution, then ask if you want
it to turn that proposal into edits to your files."
  (interactive)
  (aidermacs--send-command "/chat-mode architect")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'architect))
  (message "Switched to architect mode - aider will propose solutions before making changes"))

;;;###autoload
(defun aidermacs-switch-to-help-mode ()
  "Switch aider to help mode.
In help mode, aider will answer questions about using aider, configuring,
troubleshooting, etc."
  (interactive)
  (aidermacs--send-command "/chat-mode help")
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (setq-local aidermacs--current-mode 'help))
  (message "Switched to help mode - aider will answer questions about using aider"))

;; Initialize the cleanup mechanisms
(aidermacs--setup-ediff-cleanup-hooks)

(provide 'aidermacs)
;;; aidermacs.el ends here

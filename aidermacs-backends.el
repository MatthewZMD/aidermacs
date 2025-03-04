;;; aidermacs-backends.el --- Backend implementations for aidermacs.el -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 0.9.0
;; Keywords: ai emacs agents llm aider ai-pair-programming, convenience, tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; Originally forked from: Kang Tu <tninja@gmail.com> Aider.el
;; SPDX-License-Identifier: Apache-2.0
;;
;; This file is not part of GNU Emacs.
;;
;;; Commentary:
;;
;; This file serves as the backend dispatcher for Aidermacs,
;; allowing users to choose between different backend implementations
;; for interacting with the Aider process. Currently, it supports
;; Comint and VTerm backends.
;;
;; Key features include:
;; - Selection of the backend via the `aidermacs-backend` custom variable.
;; - Abstraction of backend-specific functions for running Aider.
;; - Management of output history and callbacks.
;;
;;; Code:

(require 'aidermacs-backend-comint)
(when (commandp 'vterm)
  (require 'aidermacs-backend-vterm))

(declare-function aidermacs-run-vterm "aidermacs-backend-vterm"
                  (program args buffer-name))
(declare-function aidermacs--send-command-vterm "aidermacs-backend-vterm"
                  (buffer command))
(declare-function aidermacs-project-root "aidermacs"
                  ())

(defgroup aidermacs-backends nil
  "Backend customization for aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-backend 'comint
  "Backend to use for the aidermacs process.
Options are `'comint' (the default) or `'vterm'.  When set to `'vterm',
aidermacs launches a fully functional vterm buffer instead
of using a comint process."
  :type '(choice (const :tag "Comint" comint)
                 (const :tag "VTerm" vterm))
  :group 'aidermacs-backends)

;; Core output management functionality
(defgroup aidermacs-output nil
  "Output handling for aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-output-limit 10
  "Maximum number of output entries to keep in history."
  :type 'integer
  :group 'aidermacs-output)

(defvar-local aidermacs--output-history nil
  "List to store aidermacs output history.
Each entry is a cons cell (timestamp . output-text).")

(defvar-local aidermacs--last-command nil
  "Store the last command sent to aidermacs.")

(defvar-local aidermacs--current-output ""
  "Accumulator for current output being captured as a string.")

(defun aidermacs-get-output-history (&optional limit)
  "Get the output history, optionally limited to LIMIT entries.
LIMIT is the maximum number of entries to return.
Returns a list of (timestamp . output-text) pairs, most recent first."
  (let ((history aidermacs--output-history))
    (if limit
        (seq-take history limit)
      history)))

(defun aidermacs-clear-output-history ()
  "Clear the output history."
  (interactive)
  (setq aidermacs--output-history nil))

(defvar-local aidermacs--current-callback nil
  "Store the callback function for the current command.")

(defvar-local aidermacs--in-callback nil
  "Flag to prevent recursive callbacks.")

(defvar-local aidermacs--tracked-files nil
  "List of files that have been mentioned in the aidermacs output.
This is used to avoid having to run /ls repeatedly.")

(defun aidermacs--verify-tracked-files ()
  "Verify if files in `aidermacs--tracked-files` exist relative to the project root.
Remove any files that don't exist."
  (let ((project-root (aidermacs-project-root))
        (valid-files nil))
    (dolist (file aidermacs--tracked-files)
      (let* ((is-readonly (string-match-p " (read-only)$" file))
             (actual-file (if is-readonly
                              (substring file 0 (- (length file) 12))
                            file))
             (full-path (expand-file-name actual-file project-root)))
        (when (file-exists-p full-path)
          (push file valid-files))))
    (setq aidermacs--tracked-files valid-files)))

(defun aidermacs--parse-output-for-files (output)
  "Parse OUTPUT for mentions of files and add them to `aidermacs--tracked-files`.
Looks for patterns like 'Applied edit to <filename>' and similar."
  (when output
    (let ((lines (split-string output "\n")))
      (dolist (line lines)
        (cond
         ;; Applied edit to <filename>
         ((string-match "Applied edit to \\(.+\\)" line)
          (when-let ((file (match-string 1 line)))
            (add-to-list 'aidermacs--tracked-files file)))

         ;; Added <filename> to the chat.
         ((string-match "Added \\(.+\\) to the chat" line)
          (when-let ((file (match-string 1 line)))
            (add-to-list 'aidermacs--tracked-files file)))

         ;; Removed <filename> from the chat
         ((string-match "Removed \\(.+\\) from the chat" line)
          (when-let ((file (match-string 1 line)))
            (setq aidermacs--tracked-files (delete file aidermacs--tracked-files))))

         ;; Added <filename> to read-only files.
         ((string-match "Added \\(.+\\) to read-only files" line)
          (when-let ((file (match-string 1 line)))
            (add-to-list 'aidermacs--tracked-files (concat file " (read-only)"))))))
      ;; Verify all tracked files exist
      (aidermacs--verify-tracked-files))))

(defun aidermacs-reset-tracked-files ()
  "Reset the list of tracked files and force a refresh."
  (interactive)
  (setq aidermacs--tracked-files nil)
  (aidermacs--get-files-in-session (lambda (files)
                                     (message "Refreshed file list: %s" files))))

(defun aidermacs--store-output (output)
  "Store output string in the history with timestamp.
OUTPUT is the string to store.
If there's a callback function, call it with the output."
  (setq aidermacs--current-output (substring-no-properties output))
  (push (cons (current-time) (substring-no-properties output)) aidermacs--output-history)
  (when (> (length aidermacs--output-history) aidermacs-output-limit)
    (setq aidermacs--output-history
          (seq-take aidermacs--output-history aidermacs-output-limit)))
  ;; Parse output for file mentions
  (aidermacs--parse-output-for-files output)
  (unless aidermacs--in-callback
    (when aidermacs--current-callback
      (let ((aidermacs--in-callback t))
        (funcall aidermacs--current-callback)
        (setq aidermacs--current-callback nil)))))

;; Backend dispatcher functions
(defun aidermacs-run-backend (program args buffer-name)
  "Run aidermacs using the selected backend.
PROGRAM is the aidermacs executable path.  ARGS are command line arguments.
BUFFER-NAME is the name for the aidermacs buffer."
  (message "Running %s with %s" program args)
  (cond
   ((eq aidermacs-backend 'vterm)
    (aidermacs-run-vterm program args buffer-name))
   (t
    (aidermacs-run-comint program args buffer-name))))

(defun aidermacs--is-aidermacs-buffer-p (&optional buffer)
  "Check if BUFFER is any type of aidermacs buffer.
If BUFFER is nil, check the current buffer.
Returns non-nil if the buffer name matches the aidermacs buffer pattern
and is using either comint or vterm mode."
  (let ((buf (or buffer (current-buffer))))
    (with-current-buffer buf
      (and (string-match-p "^\\*aidermacs:" (buffer-name buf))
           (or (derived-mode-p 'comint-mode)
               (and (fboundp 'vterm-mode)
                    (derived-mode-p 'vterm-mode)))))))

(defun aidermacs--send-command-backend (buffer command &optional redirect callback)
  "Send command to buffer using the appropriate backend.
BUFFER is the target buffer.  COMMAND is the text to send.
If REDIRECT is non-nil it redirects the output (hidden) for comint backend.
If CALLBACK is non-nil it will be called after the command finishes."
  (setq aidermacs--last-command command
        aidermacs--current-callback callback)
  (if (eq aidermacs-backend 'vterm)
      (aidermacs--send-command-vterm buffer command)
    (if redirect
        (aidermacs--send-command-redirect-comint buffer command)
      (aidermacs--send-command-comint buffer command))))

(provide 'aidermacs-backends)

;;; aidermacs-backends.el ends here

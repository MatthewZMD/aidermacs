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

(defvar aidermacs--output-history nil
  "List to store aidermacs output history.
Each entry is a cons cell (timestamp . output-text).")

(defvar aidermacs--last-command nil
  "Store the last command sent to aidermacs.")

(defvar aidermacs--current-output ""
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

(defvar aidermacs--current-callback nil
  "Store the callback function for the current command.")

(defvar aidermacs--in-callback nil
  "Flag to prevent recursive callbacks.")

(defun aidermacs--store-output (output)
  "Store output string in the history with timestamp.
OUTPUT is the string to store.
If there's a callback function, call it with the output."
  (setq aidermacs--current-output (substring-no-properties output))
  (push (cons (current-time) (substring-no-properties output)) aidermacs--output-history)
  (when (> (length aidermacs--output-history) aidermacs-output-limit)
    (setq aidermacs--output-history
          (seq-take aidermacs--output-history aidermacs-output-limit)))
  (unless aidermacs--in-callback
    (when aidermacs--current-callback
      (let ((aidermacs--in-callback t))
        (funcall aidermacs--current-callback output)
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

(defun aidermacs--send-command-backend (buffer command)
  "Send command to buffer using the appropriate backend.
BUFFER is the target buffer.  COMMAND is the text to send."
  (setq aidermacs--last-command command
        aidermacs--current-output nil)
  (if (eq aidermacs-backend 'vterm)
      (aidermacs--send-command-vterm buffer command)
    (aidermacs--send-command-comint buffer command)))

(defun aidermacs--send-command-redirect-backend (buffer command &optional callback)
  "Send command to buffer using the appropriate backend.
BUFFER is the target buffer.  COMMAND is the text to send.
CALLBACK if provided will be called with the command output when available."
  (setq aidermacs--last-command command
        aidermacs--current-output nil
        aidermacs--current-callback callback)
  (if (eq aidermacs-backend 'vterm)
      (aidermacs--send-command-vterm buffer command)
    (aidermacs--send-command-redirect-comint buffer command)))

(provide 'aidermacs-backends)

;;; aidermacs-backends.el ends here

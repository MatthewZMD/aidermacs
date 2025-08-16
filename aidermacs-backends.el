;;; aidermacs-backends.el --- Backend dispatcher for aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.5
;; Keywords: ai emacs llm aider ai-pair-programming tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; SPDX-License-Identifier: Apache-2.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Backend dispatcher for Aidermacs, allowing choice between
;; different backend implementations for Aider process.
;; Supports Comint and VTerm backends.
;;
;; Features:
;; - Backend selection via `aidermacs-backend` variable
;; - Abstracts backend-specific Aider functions
;; - Manages output history and callbacks

;;; Code:

(require 'aidermacs-backend-comint)
(require 'aidermacs-backend-vterm)

(declare-function aidermacs-run-vterm "aidermacs-backend-vterm")
(declare-function aidermacs--send-command-vterm "aidermacs-backend-vterm")
(declare-function aidermacs--prepare-for-code-edit "aidermacs-output")
(declare-function aidermacs-project-root "aidermacs")
(declare-function aidermacs--get-files-in-session "aidermacs")

(defgroup aidermacs-backends nil
  "Backend dispatcher for aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-backend 'comint
  "Backend to use for the aidermacs process.
Options are `comint' (the default) or `vterm'.  When set to `vterm',
aidermacs launches a fully functional vterm buffer instead
of using a comint process."
  :type '(choice (const :tag "Comint" comint)
                 (const :tag "VTerm" vterm)))

(defvar-local aidermacs--current-callback nil
  "Store the callback function for the current command.")

(defvar-local aidermacs--in-callback nil
  "Flag to prevent recursive callbacks.")

(defcustom aidermacs-before-run-backend-hook nil
  "Hook run before the aidermacs backend is startd."
  :type 'hook)

(defconst aidermacs-prompt-regexp "^[^[:space:]<]*>[[:space:]]+$"
  "Regexp to match Aider's command prompt.")

;; Backend dispatcher functions
(defun aidermacs-run-backend (program args buffer-name)
  "Run aidermacs using the selected backend.
PROGRAM is the aidermacs executable path.  ARGS are command line arguments.
BUFFER-NAME is the name for the aidermacs buffer."
  (message "Running %s with %s" program args)
  ;; make a copy of process-environment, so that secrets set in the hook is only visible by aider
  (let ((process-environment process-environment))
    (run-hooks 'aidermacs-before-run-backend-hook)
    (cond
     ((eq aidermacs-backend 'vterm)
      (aidermacs-run-vterm program args buffer-name))
     (t
      (aidermacs-run-comint program args buffer-name)))))

(defun aidermacs--is-aidermacs-buffer-p (&optional buffer)
  "Check if BUFFER is any type of aidermacs buffer.
If BUFFER is nil, check the current buffer.
Returns non-nil if the buffer has either `aidermacs-comint-mode' or
`aidermacs-vterm-mode' enabled."
  (let ((buf (or buffer (current-buffer))))
    (with-current-buffer buf
      (or (derived-mode-p 'aidermacs-comint-mode)
          (bound-and-true-p aidermacs-vterm-mode)))))

(defun aidermacs--send-command-backend (buffer command &optional redirect)
  "Send command to buffer using the appropriate backend.
BUFFER is the target buffer.  COMMAND is the text to send.
If REDIRECT is non-nil it redirects the output (hidden) for comint backend."
  (if (eq aidermacs-backend 'vterm)
      (aidermacs--send-command-vterm buffer command)
    (if redirect
        (aidermacs--send-command-redirect-comint buffer command)
      (aidermacs--send-command-comint buffer command))))

(provide 'aidermacs-backends)

;;; aidermacs-backends.el ends here

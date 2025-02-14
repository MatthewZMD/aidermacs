;;; aidermacs-backends.el --- Backend implementations for aidermacs.el -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 0.5.0
;; Package-Requires: ((emacs "26.1") (transient "0.3.0"))
;; Keywords: ai emacs agents llm aider ai-pair-programming, convenience, tools
;; URL: https://github.com/MatthewZMD/aidermacs.el
;; Originally forked from: Kang Tu <tninja@gmail.com> Aider.el

;;; Commentary:
;; Backend dispatcher for aidermacs.el

;;; Code:

(require 'aidermacs-backend-comint)
(when (require 'vterm nil 'noerror)
  (require 'aidermacs-backend-vterm))

(defgroup aidermacs-backends nil
  "Backend customization for aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-backend 'comint
  "Backend to use for the aidermacs process.
Options are 'comint (the default) or 'vterm. When set to 'vterm, aidermacs will
launch a fully functional vterm buffer (with bracketed paste support) instead
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
  "Store OUTPUT string in the history with timestamp.
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
PROGRAM is the aidermacs executable path, ARGS are command line arguments,
and BUFFER-NAME is the name for the aidermacs buffer."
  (cond
   ((eq aidermacs-backend 'vterm)
    (aidermacs-run-vterm program args buffer-name))
   (t
    (aidermacs-run-comint program args buffer-name))))

(defun aidermacs--send-command-backend (buffer command)
  "Send COMMAND to BUFFER using the appropriate backend.
CALLBACK if provided will be called with the command output when available."
  (setq aidermacs--last-command command
        aidermacs--current-output nil
        aidermacs--current-callback nil)
  (if (eq aidermacs-backend 'vterm)
      (aidermacs--send-command-vterm buffer command)
    (aidermacs--send-command-comint buffer command)))

(defun aidermacs--redirect-send-command-backend (buffer command &optional callback)
  "Send COMMAND to BUFFER using the appropriate backend.
CALLBACK if provided will be called with the command output when available."
  (setq aidermacs--last-command command
        aidermacs--current-output nil
        aidermacs--current-callback callback)
  (if (eq aidermacs-backend 'vterm)
      (aidermacs--send-command-vterm buffer command)
    (aidermacs--redirect-send-command-comint buffer command)))

(provide 'aidermacs-backends)

;;; aidermacs-backends.el ends here

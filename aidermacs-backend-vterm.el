;;; aidermacs-backend-vterm.el --- VTerm backend for aidermacs.el -*- lexical-binding: t; -*-
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
;; This file implements the VTerm backend for Aidermacs, providing
;; an interface for interacting with the Aider process within a
;; VTerm buffer. It leverages the VTerm package for a more
;; feature-rich terminal experience.
;;
;; Key features include:
;; - Integration with the VTerm package for terminal emulation.
;; - Asynchronous output processing using timers and advices.
;; - Custom keybindings for multi-line input.
;; - Management of the Aider process within a VTerm buffer.
;;
;;; Code:

(require 'vterm nil 'noerror)
(require 'cl-lib)

;; we want to ensure these two variables are dynamic binding
(defvar vterm-shell)
(defvar vterm-buffer-name)

;; Forward declaration to avoid compiler warnings
(declare-function vterm--render "vterm")
(declare-function vterm--get-prompt-point "vterm")
(declare-function vterm-other-window "vterm")
(declare-function vterm-send-string "vterm")
(declare-function vterm-send-return "vterm")
(declare-function vterm-insert "vterm")


(defvar-local aidermacs--vterm-active-timer nil
  "Store the active timer for vterm output processing.")

(defvar-local aidermacs--vterm-last-check-point nil
  "Store the last position checked in the vterm buffer.")


(defvar-local aidermacs-vterm-check-interval 0.2
  "Interval in seconds between checks for command completion in vterm.")


(defcustom aidermacs-vterm-multiline-newline-key "S-<return>"
  "Key binding to enter a newline without sending in vterm."
  :type 'string
  :group 'aidermacs)

(defun aidermacs--vterm-check-finish-sequence-repeated (proc orig-filter start-point expected)
  "Check for the finish sequence in PROC's buffer.
PROC is the process to check.  ORIG-FILTER is the original process filter.
START-POINT is the starting position for output capture.  EXPECTED is the
pattern to match.  If the finish sequence is detected, store the output via
`aidermacs--store-output`, restore ORIG-FILTER, and return t."
  (when (buffer-live-p (process-buffer proc))
    (with-current-buffer (process-buffer proc)
      ;; Render vterm if the function is available
      (when (fboundp 'vterm--render)
        (condition-case nil
            (vterm--render)
          (error nil)))
      ;; Get current prompt point
      (let* ((prompt-point (condition-case nil
                               (vterm--get-prompt-point)
                             (error (point-max))))
             ;; Only check if we have a new prompt or haven't checked this position yet
             (last-check (or aidermacs--vterm-last-check-point start-point))
             (should-check (> prompt-point last-check)))

        ;; Update the last check point
        (setq aidermacs--vterm-last-check-point prompt-point)

        (when should-check
          (let* ((seq-start (or (save-excursion
                                  (goto-char prompt-point)
                                  (condition-case nil
                                      (search-backward "\n" nil t)
                                    (error nil)))
                                (point-min)))
                 ;; Only get the prompt line, not the whole sequence (limit to 200 chars)
                 (prompt-line-end (min (+ seq-start 200) (point-max)))
                 (prompt-line (buffer-substring-no-properties seq-start prompt-line-end)))

            ;; If we found a shell prompt
            (when (string-match-p expected prompt-line)
              (let ((output (buffer-substring-no-properties start-point seq-start)))
                (aidermacs--store-output (string-trim output)))
              (set-process-filter proc orig-filter))))))))

(defun aidermacs--vterm-output-advice (orig-fun &rest args)
  "Capture vterm output until the finish sequence appears.
ORIG-FUN is the original function being advised.  ARGS are its arguments.
This sets a temporary process filter that checks for the finish sequence
after each output chunk, reducing the need for timers."
  (if (aidermacs--is-aidermacs-buffer-p)
      (let* ((start-point (condition-case nil
                              (vterm--get-prompt-point)
                            (error (point-min))))
             (proc (get-buffer-process (current-buffer)))
             ;; Simplified pattern that just looks for a shell prompt
             (expected "^[^[:space:]]*>[[:space:]]")
             (orig-filter (process-filter proc)))

        ;; Initialize tracking variables
        (setq-local aidermacs--vterm-last-check-point nil)

        ;; Set our temporary process filter
        (set-process-filter
         proc
         (lambda (proc string)
           ;; Call the original filter
           (funcall orig-filter proc string)

           ;; Only set up a timer if we don't already have one
           (unless aidermacs--vterm-active-timer
             ;; Set up a timer to check for completion
             (setq aidermacs--vterm-active-timer
                   (run-with-timer
                    aidermacs-vterm-check-interval
                    aidermacs-vterm-check-interval
                    (lambda ()
                      (when (aidermacs--vterm-check-finish-sequence-repeated
                             proc orig-filter start-point expected)
                        (when (timerp aidermacs--vterm-active-timer)
                          (cancel-timer aidermacs--vterm-active-timer)
                          (setq aidermacs--vterm-active-timer nil))
                        (set-process-filter proc orig-filter))))))))
        (apply orig-fun args))
    (apply orig-fun args)))

(defun aidermacs-run-vterm (program args buffer-name)
  "Create a vterm-based buffer and run aidermacs program.
PROGRAM is the command to run.  ARGS is a list of command line arguments.
BUFFER-NAME is the name for the vterm buffer."
  (unless (require 'vterm nil t)
    (error "Vterm package is not available"))
  (unless (get-buffer buffer-name)
    (let* ((mode (if (eq (frame-parameter nil 'background-mode) 'dark)
                     "--dark-mode"
                   "--light-mode"))
           (cmd (mapconcat #'identity (append (list program mode) args) " "))
           (vterm-buffer-name buffer-name)
           (vterm-shell cmd))
      (with-current-buffer (vterm-other-window)
        (setq-local vterm-max-scrollback 1000
                    aidermacs--vterm-active-timer nil
                    aidermacs--vterm-last-check-point nil)
        (advice-add 'vterm-send-return :around #'aidermacs--vterm-output-advice)
        ;; Set up multi-line key binding
        (let ((map (make-sparse-keymap)))
          (set-keymap-parent map (current-local-map))
          (define-key map (kbd aidermacs-vterm-multiline-newline-key) #'aidermacs-vterm-insert-newline)
          (use-local-map map))
        ;; Add cleanup hook
        (add-hook 'kill-buffer-hook #'aidermacs--vterm-cleanup nil t))))
  buffer-name)

(defun aidermacs--send-command-vterm (buffer command)
  "Send command to the aidermacs vterm buffer.
BUFFER is the target buffer to send to.  COMMAND is the text to send."
  (with-current-buffer buffer
    ;; Cancel any existing timer to prevent resource leaks
    (when aidermacs--vterm-active-timer
      (cancel-timer aidermacs--vterm-active-timer)
      (setq aidermacs--vterm-active-timer nil))
    (vterm-send-string command)
    (vterm-send-return)))

(defun aidermacs-vterm-insert-newline ()
  "Insert a newline in vterm without sending the command."
  (interactive)
  (vterm-insert "\n"))

(defun aidermacs--vterm-cleanup ()
  "Clean up vterm resources when buffer is killed."
  (when aidermacs--vterm-active-timer
    (cancel-timer aidermacs--vterm-active-timer)
    (setq-local aidermacs--vterm-active-timer nil))
  (setq-local aidermacs--vterm-last-check-point nil))

(provide 'aidermacs-backend-vterm)

;;; aidermacs-backend-vterm.el ends here

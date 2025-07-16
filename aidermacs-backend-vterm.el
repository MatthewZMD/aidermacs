;;; aidermacs-backend-vterm.el --- VTerm backend for aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.5
;; Keywords: ai emacs llm aider ai-pair-programming tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; SPDX-License-Identifier: Apache-2.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Implements VTerm backend for Aidermacs, providing interface to
;; interact with Aider process in VTerm buffer. Uses VTerm package
;; for rich terminal experience.
;;
;; Features:
;; - VTerm integration for terminal emulation
;; - Async output processing with timers
;; - Custom multi-line input keybindings
;; - Aider process management in VTerm

;;; Code:

(require 'vterm nil 'noerror)
(require 'cl-lib)

;; we want to ensure these two variables are dynamic binding
(defvar vterm-shell)
(defvar vterm-buffer-name)
(defvar vterm-kill-buffer-on-exit)

;; Forward declaration to avoid compiler warnings
(declare-function vterm--render "vterm")
(declare-function vterm--get-prompt-point "vterm")
(declare-function vterm-other-window "vterm")
(declare-function vterm-send-string "vterm")
(declare-function vterm-send-return "vterm")
(declare-function vterm-insert "vterm")
(declare-function vterm-send-C-c "vterm")

(declare-function aidermacs--detect-edited-files "aidermacs")
(declare-function aidermacs--command-may-edit-files "aidermacs")
(declare-function aidermacs--is-aidermacs-buffer-p "aidermacs")
(declare-function aidermacs-get-buffer-name "aidermacs")
(declare-function aidermacs--store-output "aidermacs-output")
(declare-function aidermacs--prepare-for-code-edit "aidermacs-output")
(declare-function aidermacs--parse-output-for-files "aidermacs-output")
(declare-function aidermacs--show-ediff-for-edited-files "aidermacs-output")
(declare-function aidermacs--cleanup-temp-buffers "aidermacs-output")

(declare-function evil-define-minor-mode-key "evil-core")

(defvar aidermacs-prompt-regexp)
(defvar aidermacs--last-command)

(defgroup aidermacs-backend-vterm nil
  "VTerm backend for Aidermacs."
  :group 'aidermacs)

(defvar-local aidermacs--vterm-active-timer nil
  "Store the active timer for vterm output processing.")

(defvar-local aidermacs--vterm-last-check-point nil
  "Store the last position checked in the vterm buffer.")

(defvar-local aidermacs-vterm-check-interval 0.7
  "Interval in seconds between checks for command completion in vterm.")

(defcustom aidermacs-vterm-multiline-newline-key "S-<return>"
  "Key binding to enter a newline without sending in vterm."
  :type 'string)

(defun aidermacs--vterm-check-finish-sequence-repeated (proc orig-filter start-point)
  "Check for the finish sequence in PROC's buffer.
PROC is the process to check.  ORIG-FILTER is the original process filter.
START-POINT is the starting position for output capture.
If the finish sequence is detected, store the output via
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
             (should-check (> prompt-point last-check))
             ;; Pattern that looks for a shell prompt
             (expected aidermacs-prompt-regexp))
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
                 (prompt-line (buffer-substring-no-properties seq-start prompt-line-end))
                 (output (buffer-substring-no-properties start-point seq-start)))
            ;; Parse output for files
            (aidermacs--parse-output-for-files output)
            ;; If we found a shell prompt indicating output finished
            (when (string-match-p expected prompt-line)
              (aidermacs--store-output (string-trim output))
              (setq-local aidermacs--ready t)
              (let ((edited-files (aidermacs--detect-edited-files)))
                ;; Check if any files were edited and show ediff if needed
                (if edited-files
                    (aidermacs--show-ediff-for-edited-files edited-files)
                  (aidermacs--cleanup-temp-buffers))
                ;; Restore the original process filter now that we've finished processing
                ;; this command's output. This returns vterm to its normal behavior.
                (set-process-filter proc orig-filter)
                (aidermacs--maybe-cancel-active-timer (process-buffer proc))))))))))

(defun aidermacs--vterm-capture-output ()
  "Capture vterm output until the finish sequence appears.
ORIG-FUN is the original function being advised.  ARGS are its arguments.
This sets a temporary process filter that checks for the finish sequence
after each output chunk, reducing the need for timers."
  (when (and (aidermacs--is-aidermacs-buffer-p)
             (not (string-empty-p aidermacs--last-command)))
    (let* ((start-point (condition-case nil
                            (vterm--get-prompt-point)
                          (error (point-min))))
           (proc (get-buffer-process (current-buffer)))
           (orig-filter (process-filter proc)))
      ;; Store the command for tracking in the correct buffer
      (with-current-buffer (process-buffer proc)
        ;; Initialize tracking variables
        (setq aidermacs--vterm-last-check-point nil)
        ;; Cancel any existing timer first
        (aidermacs--maybe-cancel-active-timer)
        ;; Create a new timer immediately to start checking for command completion
        (setq aidermacs--vterm-active-timer
              (run-with-timer
               aidermacs-vterm-check-interval
               aidermacs-vterm-check-interval
               (lambda () (aidermacs--vterm-check-finish-sequence-repeated proc orig-filter start-point))))))))

(defun aidermacs--maybe-cancel-active-timer (&optional buffer)
  "Cancel the active timer if it exists.
Use BUFFER if provided, otherwise retrieve it from `aidermacs-get-buffer-name'."
  (when-let* ((buf (get-buffer (or buffer (aidermacs-get-buffer-name)))))
    (with-current-buffer buf
      (when (timerp aidermacs--vterm-active-timer)
        (cancel-timer aidermacs--vterm-active-timer)
        (setq aidermacs--vterm-active-timer nil)))))


(defun aidermacs--vterm-filter-buffer-substring (orig-fun &rest args)
  "Filter text using ORIG-FUN with ARGS for cleaner vterm display.
Converts multiple spaces to newlines and trims trailing whitespace."
  (let* ((text (apply orig-fun args))
         (space-fixed (replace-regexp-in-string "[ \t]\\{3\\}" "\n" text))
         (lines (split-string space-fixed "\n"))
         (clean-lines (mapcar #'string-trim-right lines))
         (fixed-text (string-join clean-lines "\n")))
    fixed-text))

(defcustom aidermacs-vterm-use-theme-colors t
  "Whether to use Emacs theme colors for aider.
Has effect only when using the vterm backend."
  :type 'boolean)

(defcustom aidermacs-vterm-theme-foreground-colors-plist
  '("--user-input-color" font-lock-function-name-face
    "--assistant-output-color" default
    "--tool-output-color" default
    "--tool-error-color" error
    "--tool-warning-color"  warning
    "--completion-menu-color" default
    "--completion-menu-current-color" highlight)
  "Emacs faces to use for aider color flags.
Keys are the commandline arguments to send to aider. Values are either faces,
or strings containing color codes like \"#00ccff\".

For any face specified, the :foreground attribute will be used to determine the
color. If that is unspecified, this falls back to the default face foreground
color."
  :type 'plist)

(defcustom aidermacs-vterm-theme-background-colors-plist
  '("--completion-menu-bg-color" default
    "--completion-menu-current-bg-color" highlight)
  "Emacs faces to use for aider color flags.
Keys are the commandline arguments to send to aider. Values are either faces,
or strings containing color codes like \"#00ccff\".

For any face specified, the :background attribute will be used to determine the
color. If that is unspecified, this falls back to the default face foreground
color."
  :type 'plist)

(defun aidermacs--vterm-colorname-to-rgb (name attr)
  "Convert Emacs color name NAME to RGB values with ATTR."
  (if-let* ((colors (color-values name)))
      (concat "#"
              (mapconcat (lambda (c) (format "%02X" (/ c 256))) colors))
    (face-attribute 'default attr)))

(defun aidermacs--vterm-get-face-color (face attr)
  "Return an RGB color code for FACE attribute ATTR.
If unspecified, return ATTR from default face."
  (let ((color-string (face-attribute-specified-or
		               (face-attribute face attr)
		               (face-attribute 'default attr))))
    (if (string-prefix-p "#" color-string)
        color-string
      (aidermacs--vterm-colorname-to-rgb color-string attr))))

(defun aidermacs--vterm-convert-color-arg (attr el)
  "Maps element EL from color plist, to actual RGB Values, using attribute ATTR."
  (shell-quote-argument
   (cond ((stringp el) el)
         ((facep el) (aidermacs--vterm-get-face-color el attr))
         (t (error "Invalid face or colour value: %s" el)))))

(defun aidermacs--vterm-theme-args ()
  "Create arguments for aider to try and match the current Emacs theme.
See `aidermacs-theme-colors-plist'."
  (if aidermacs-vterm-use-theme-colors
      (nconc
       (mapcar (apply-partially #'aidermacs--vterm-convert-color-arg :foreground)
               aidermacs-vterm-theme-foreground-colors-plist)
       (mapcar (apply-partially #'aidermacs--vterm-convert-color-arg :background)
               aidermacs-vterm-theme-background-colors-plist))
    (if (eq (frame-parameter nil 'background-mode) 'dark)
        '("--dark-mode")
      '("--light-mode"))))

(defun aidermacs-run-vterm (program args buffer-name)
  "Create a vterm-based buffer and run aidermacs program.
PROGRAM is the command to run.  ARGS is a list of command line arguments.
BUFFER-NAME is the name for the vterm buffer."
  (unless (require 'vterm nil t)
    (error "Vterm package is not available"))
  (unless (get-buffer buffer-name)
    (let* ((cmd (mapconcat #'identity (append `(,program ,@(aidermacs--vterm-theme-args)) args) " "))
           (vterm-buffer-name buffer-name)
           (vterm-shell cmd)
           (vterm-kill-buffer-on-exit nil))
      (with-current-buffer (vterm-other-window)
        (setq-local vterm-max-scrollback 1000
                    aidermacs--vterm-active-timer nil
                    aidermacs--vterm-last-check-point nil
                    aidermacs--ready t)
        (aidermacs-vterm-mode 1)
        ;; Add cleanup hook
        (add-hook 'kill-buffer-hook #'aidermacs--vterm-cleanup nil t))))
  buffer-name)

(defun aidermacs--send-command-vterm (buffer command)
  "Send command to the aidermacs vterm buffer.
BUFFER is the target buffer to send to.  COMMAND is the text to send."
  (with-current-buffer buffer
    (setq-local aidermacs--ready nil)
    ;; Cancel any existing timer to prevent resource leaks
    (aidermacs--maybe-cancel-active-timer)
    ;; Only process if we have a non-empty command
    (when (and command (not (string-empty-p (string-trim command))))
      ;; Store the command and reset output
      (setq-local aidermacs--last-command command)
      ;; Send the command
      (vterm-send-string command)
      (aidermacs-vterm-send-return))))

(defun aidermacs-vterm-send-return ()
  "Send return to the aidermacs vterm buffer, and process the output."
  (interactive)
  (aidermacs--vterm-capture-keyboard-input)
  (aidermacs--vterm-capture-output)
  (vterm-send-return))

(defun aidermacs-vterm-send-C-c ()
  "Send interrupt to the aidermacs vterm buffer, and cleanup."
  (interactive)
  (vterm-send-C-c)
  (aidermacs--cleanup-temp-files-on-interrupt-vterm))

(defun aidermacs-vterm-insert-newline ()
  "Insert a newline in vterm without sending the command."
  (interactive)
  (vterm-insert "\n"))

(defun aidermacs--vterm-capture-keyboard-input ()
  "Capture keyboard input in vterm."
  (when (aidermacs--is-aidermacs-buffer-p)
    ;; Get the current line content which should be the command
    ;; TODO: current line may not be enough
    (save-excursion
      (let* ((prompt-point (condition-case nil
                               (vterm--get-prompt-point)
                             (error (point-min))))
             (command (buffer-substring-no-properties
                       prompt-point
                       (line-end-position))))
        (when (not (string-empty-p command))
          (setq-local aidermacs--last-command command)
          (when (aidermacs--command-may-edit-files command)
            (aidermacs--prepare-for-code-edit)))))))

(defun aidermacs--vterm-cleanup ()
  "Clean up vterm resources when buffer is killed."
  (aidermacs--maybe-cancel-active-timer)
  (setq-local aidermacs--vterm-last-check-point nil))

(defun aidermacs--cleanup-temp-files-on-interrupt-vterm ()
  "Run `aidermacs--cleanup-temp-buffers' after interrupting a vterm subjob.
_ARGS are the arguments."
  (when (aidermacs--is-aidermacs-buffer-p)
    ;; Cancel any active timer
    (aidermacs--maybe-cancel-active-timer)
    (aidermacs--cleanup-temp-buffers)))

(defvar aidermacs-vterm-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd aidermacs-vterm-multiline-newline-key) #'aidermacs-vterm-insert-newline)
    (define-key map (kbd "RET") #'aidermacs-vterm-send-return)
    (define-key map (kbd "<return>") #'aidermacs-vterm-send-return)
    (define-key map (kbd "C-c C-c") #'aidermacs-vterm-send-C-c)
    (when (featurep 'evil)
      (evil-define-minor-mode-key map 'insert
                                  (kbd "RET") (function aidermacs-vterm-send-return)))
    map)
  "Keymap used when `aidermacs-vterm-mode' is enabled.")

(define-minor-mode aidermacs-vterm-mode
  "Minor mode for vterm backend buffer used by aidermacs."
  :init-value nil
  :keymap aidermacs-vterm-mode-map
  (add-function :around (local 'filter-buffer-substring-function)
                #'aidermacs--vterm-filter-buffer-substring))

(provide 'aidermacs-backend-vterm)
;;; aidermacs-backend-vterm.el ends here

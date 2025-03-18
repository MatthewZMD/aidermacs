;;; aidermacs-backend-comint.el --- Comint backend for aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.0
;; Keywords: ai emacs llm aider ai-pair-programming tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; SPDX-License-Identifier: Apache-2.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;;
;; Implements Comint backend for Aidermacs, providing interface to
;; interact with Aider process in Emacs buffer. Handles command
;; sending, output display, and interaction flow.
;;
;; Features:
;; - Syntax highlighting for Aider output
;; - Custom keybindings for multi-line input
;; - Process management in Comint buffer

;; Originally forked from: Kang Tu <tninja@gmail.com> Aider.el

;;; Code:

(require 'comint)
(require 'cl-lib)
(require 'map)

;; Forward declarations
(declare-function aidermacs--prepare-for-code-edit "aidermacs-output")
(declare-function aidermacs--process-message-if-multi-line "aidermacs")
(declare-function aidermacs--command-may-edit-files "aidermacs")
(declare-function aidermacs--store-output "aidermacs-output")
(declare-function aidermacs--is-aidermacs-buffer-p "aidermacs-backends")
(declare-function aidermacs--parse-output-for-files "aidermacs-output")
(declare-function aidermacs--show-ediff-for-edited-files "aidermacs-output")
(declare-function aidermacs--cleanup-temp-buffers "aidermacs-output")
(declare-function aidermacs--detect-edited-files "aidermacs-output")

(defvar aidermacs--last-command)

(defgroup aidermacs-backend-comint nil
  "Comint backend for Aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-language-name-map '(("elisp" . "emacs-lisp")
                                         ("bash" . "sh")
                                         ("objective-c" . "objc")
                                         ("objectivec" . "objc")
                                         ("cpp" . "c++"))
  "Map external language names to Emacs names."
  :type '(alist :key-type (string :tag "Language Name/Alias")
                :value-type (string :tag "Mode Name (without -mode)")))

;; FIXME: Hmm... seems to use standard diff3 markers.  Maybe some code
;; in `smerge-mode.el' could be (re)used?
(defconst aidermacs-search-marker "<<<<<<< SEARCH")
(defconst aidermacs-diff-marker "=======")
(defconst aidermacs-replace-marker ">>>>>>> REPLACE")
(defconst aidermacs-fence-marker "```")
(defvar aidermacs-block-re
  (format "^\\(?:\\(?1:%s\\|%s\\)\\|\\(?1:%s\\).+\\)$" aidermacs-search-marker aidermacs-diff-marker aidermacs-fence-marker))

(defcustom aidermacs-comint-multiline-newline-key "S-<return>"
  "Key binding for `comint-accumulate' in Aidermacs buffers.
This allows for multi-line input without sending the command."
  :type 'string)

(defface aidermacs-command-separator
  '((((type graphic)) :strike-through t :extend t)
    (((type tty)) :inherit font-lock-comment-face :underline t :extend t))
  "Face for command separator in aidermacs.")

(defface aidermacs-command-text
  '((t :inherit bold))
  "Face for commands sent to aidermacs buffer.")

(defface aidermacs-search-replace-block
  '((t :inherit diff-refine-added :bold t))
  "Face for search/replace block content.")

(defvar aidermacs-font-lock-keywords
  '(("^\x2500+\n?" 0 '(face aidermacs-command-separator) t)
    ("^\x2500+" 0 '(face nil display (space :width 2)))
    ("^\\([0-9]+\\). " 0 font-lock-constant-face)
    ("^>>>>>>> REPLACE" 0 'aidermacs-search-replace-block t)
    ("^<<<<<<< SEARCH" 0 'aidermacs-search-replace-block t)
    ("^[:space:]*\\(```\\)\\([^[:space:]]*\\)" (1 'shadow t) (2 font-lock-builtin-face t))
    ("^=======$" 0 'aidermacs-search-replace-block t))
  "Font lock keywords for aidermacs buffer.")

;; Buffer-local variables for syntax highlighting state
(defvar-local aidermacs--syntax-block-delimiter nil
  "The expected delimiter (diff/replace/fence) for the current syntax block.")

(defvar-local aidermacs--syntax-block-start-pos nil
  "The starting position of the current syntax block.")

(defvar-local aidermacs--syntax-block-end-pos nil
  "The end position of the current syntax block.")

(defvar-local aidermacs--syntax-last-output-pos nil
  "Position tracker for incremental syntax highlighting.")

(defvar-local aidermacs--syntax-work-buffer nil
  "Temporary buffer used for syntax highlighting operations.")

(defvar-local aidermacs--syntax-block-marker nil
  "Store the current block marker (SEARCH/REPLACE/fence) being processed.
This variable holds the actual marker text (e.g., <<<<<<< SEARCH, =======, etc.)
that was matched at the start of the current syntax block.")

(defvar-local aidermacs--comint-output-temp ""
  "Temporary output variable storing the raw output string.")

(defvar aidermacs-prompt-regexp)

(defun aidermacs--comint-output-filter (output)
  "Accumulate OUTPUT string until a prompt is detected, then store it."
  (when (and (aidermacs--is-aidermacs-buffer-p) (not (string-empty-p output)))
    (setq aidermacs--comint-output-temp
          (concat aidermacs--comint-output-temp (substring-no-properties output)))
    ;; Check if the output contains a prompt
    (when (string-match-p aidermacs-prompt-regexp aidermacs--comint-output-temp)
      (aidermacs--store-output aidermacs--comint-output-temp)
      ;; Check if any files were edited and show ediff if needed
      (let ((edited-files (aidermacs--detect-edited-files)))
        (if edited-files
            (aidermacs--show-ediff-for-edited-files edited-files)
          (aidermacs--cleanup-temp-buffers)))
      (setq aidermacs--comint-output-temp ""))))

(defun aidermacs-reset-font-lock-state ()
  "Reset font lock state to default for processing a new source block."
  (setq aidermacs--syntax-block-delimiter nil
        aidermacs--syntax-last-output-pos nil
        aidermacs--syntax-block-start-pos nil
        aidermacs--syntax-block-end-pos nil
        aidermacs--syntax-block-marker nil))

(defun aidermacs-fontify-blocks (_output)
  "Fontify search/replace blocks in comint output.
_OUTPUT is the text to be processed."
  (save-excursion
    (goto-char (or aidermacs--syntax-last-output-pos
                   comint-last-output-start))
    (beginning-of-line)

    ;; Continue processing existing block if we're in one
    (when aidermacs--syntax-block-start-pos
      (aidermacs--fontify-block))

    (setq aidermacs--syntax-last-output-pos nil)
    ;; Look for new blocks if we're not in one
    (while (and (null aidermacs--syntax-block-start-pos)
                (null aidermacs--syntax-last-output-pos)
                (re-search-forward aidermacs-block-re nil t))

      ;; If it is code fence marker, we need to check if there is a SEARCH marker
      ;; directly after it
      (when (equal (match-string 1) aidermacs-fence-marker)
        (let* ((next-line (min (point-max) (1+ (line-end-position))))
               (line-text (buffer-substring
                           next-line
                           (min (point-max) (+ next-line (length aidermacs-search-marker))))))
          (cond ((equal line-text aidermacs-search-marker)
                 ;; Next line is a SEARCH marker. use that instead of the fence marker
                 (re-search-forward (format "^\\(%s\\)" aidermacs-search-marker) nil t))
                ((string-prefix-p line-text aidermacs-search-marker)
                 ;; Next line *might* be a SEARCH marker. Don't process more of
                 ;; the buffer until we know for sure
                 (setq aidermacs--syntax-last-output-pos comint-last-output-start)))))

      (unless aidermacs--syntax-last-output-pos
        ;; Set up new block state
        (setq aidermacs--syntax-block-marker (match-string 1))
        (setq aidermacs--syntax-block-start-pos (line-end-position)
              aidermacs--syntax-block-end-pos (line-end-position)
              aidermacs--syntax-block-delimiter
              (pcase aidermacs--syntax-block-marker
                ((pred (equal aidermacs-search-marker)) aidermacs-diff-marker)
                ((pred (equal aidermacs-diff-marker)) aidermacs-replace-marker)
                ((pred (equal aidermacs-fence-marker)) aidermacs-fence-marker)))

        (with-current-buffer aidermacs--syntax-work-buffer
          (erase-buffer))

        ;; Set the major-mode of the font lock buffer unless this is the second half of
        ;; SEARCH/REPLACE block. In that case reuse the previous mode
        (unless (equal aidermacs--syntax-block-marker aidermacs-diff-marker)
          (let ((mode (aidermacs--guess-major-mode)))
            (with-current-buffer aidermacs--syntax-work-buffer
              (unless (eq mode major-mode)
                (condition-case e
                    (let ((inhibit-message t))
                      (funcall mode))
                  (error "aidermacs: Failed to init major-mode `%s' for font-locking: %s" mode e))))))

        ;; Process initial content
        (aidermacs--fontify-block)))))

(defun aidermacs--fontify-block ()
  "Fontify as much of the current source block as possible."
  (let* ((last-bol (save-excursion
                     (goto-char (point-max))
                     (line-beginning-position)))
         (last-output-start aidermacs--syntax-block-end-pos)
         end-of-block-p)

    (setq aidermacs--syntax-block-end-pos
          (cond ((re-search-forward (concat "^" aidermacs--syntax-block-delimiter "$") nil t)
                 ;; Found the end of the block
                 (setq end-of-block-p t)
                 (line-beginning-position))
                ((string-prefix-p (buffer-substring last-bol (point-max)) aidermacs--syntax-block-delimiter)
                 ;; The end of the text *might* be the end marker. back up to
                 ;; make sure we don't process it until we know for sure
                 last-bol)
                ;; We can process till the end of the text
                (t (point-max))))

    ;; Append new content to temp buffer and fontify
    (let ((new-content (buffer-substring-no-properties
                        last-output-start
                        aidermacs--syntax-block-end-pos))
          (pos aidermacs--syntax-block-start-pos)
          (font-pos 0)
          fontified)

      ;; Insert the new text and get the fontified result
      (with-current-buffer aidermacs--syntax-work-buffer
        (goto-char (point-max))
        (insert new-content)
        (with-demoted-errors "aidermacs block font lock error: %s"
          (let ((inhibit-message t))
            (font-lock-ensure)))
        (setq fontified (buffer-string)))

      ;; Apply the faces to the buffer
      (remove-overlays aidermacs--syntax-block-start-pos aidermacs--syntax-block-end-pos)
      (while (< pos aidermacs--syntax-block-end-pos)
        (let* ((next-font-pos (or (next-property-change font-pos fontified) (length fontified)))
               (next-pos (+ aidermacs--syntax-block-start-pos next-font-pos))
               (face (get-text-property font-pos 'face fontified)))
          (ansi-color-apply-overlay-face pos next-pos face)
          (setq pos next-pos
                font-pos next-font-pos))))

    ;; If we found the end marker, finalize the block
    (when end-of-block-p
      (when (equal aidermacs--syntax-block-delimiter aidermacs-diff-marker)
        ;; we will need to process the other half of the SEARCH/REPLACE block.
        ;; Backup so it will get matched
        (beginning-of-line))
      (aidermacs-reset-font-lock-state))))

(defun aidermacs--guess-major-mode ()
  "Extract the major mode from fence markers or filename."
  (or
   ;; check if the block has a language id
   (when (save-excursion
           (end-of-line)
           (re-search-backward "^[:space:]*``` *\\([^[:space:]]+\\)" (line-beginning-position -1) t))
     (let* ((lang (downcase (match-string 1)))
            (mode (map-elt aidermacs-language-name-map lang lang)))
       (intern-soft (concat mode "-mode"))))
   ;; check the file extension in auto-mode-alist
   (when (save-excursion
           (or (re-search-backward "[fF]ile: *\\([^*[:space:]]+\\)" (line-beginning-position -3) t)
               (re-search-backward "^\\([^`[:space:]]+\\)$" (line-beginning-position -3) t)))
     (let ((file (match-string 1)))
       (cdr (cl-assoc-if (lambda (re) (string-match re file)) auto-mode-alist))))
   (progn
     (message "aidermacs warning: can't detect major mode at %d" (point))
     'fundamental-mode)))

(defun aidermacs--comint-cleanup-hook ()
  "Clean up the fontify buffer."
  (when (bufferp aidermacs--syntax-work-buffer)
    (kill-buffer aidermacs--syntax-work-buffer)))

(defun aidermacs-input-sender (proc string)
  "Reset font-lock state before executing a command.
PROC is the process to send to.  STRING is the command to send."
  (aidermacs-reset-font-lock-state)
  ;; Store the command for tracking in the correct buffer
  (with-current-buffer (process-buffer proc)
    (if (member (downcase string) '("" "y" "n" "d" "yes" "no"))
        (aidermacs--parse-output-for-files aidermacs--comint-output-temp)
      (setq aidermacs--last-command string)
      (when (aidermacs--command-may-edit-files string)
        (aidermacs--prepare-for-code-edit))))
  (comint-simple-send proc (aidermacs--process-message-if-multi-line string)))

(defun aidermacs-run-comint (program args buffer-name)
  "Create a comint-based buffer and run aidermacs program.
PROGRAM is the executable path.  ARGS are command line arguments.
BUFFER-NAME is the name for the aidermacs buffer."
  (let ((comint-terminfo-terminal "eterm-color")
        (args (append args (list "--no-pretty" "--no-fancy-input"))))
    (unless (comint-check-proc buffer-name)
      (apply #'make-comint-in-buffer "aidermacs" buffer-name program nil args)
      (with-current-buffer buffer-name
        (comint-mode)
        (setq-local comint-prompt-regexp aidermacs-prompt-regexp)
        (setq-local comint-input-sender 'aidermacs-input-sender)
        (setq aidermacs--syntax-work-buffer
              (get-buffer-create (concat " *aidermacs-syntax" buffer-name)))
        (add-hook 'kill-buffer-hook #'aidermacs--comint-cleanup-hook nil t)
        (add-hook 'comint-output-filter-functions #'aidermacs-fontify-blocks 100 t)
        (add-hook 'comint-output-filter-functions #'aidermacs--comint-output-filter)
        (aidermacs-comint-mode 1)
        (font-lock-add-keywords nil aidermacs-font-lock-keywords t)))))

(defun aidermacs--send-command-comint (buffer command)
  "Send command to the aidermacs comint buffer.
BUFFER is the target buffer.  COMMAND is the text to send."
  (with-current-buffer buffer
    (let ((process (get-buffer-process buffer))
          (inhibit-read-only t))
      (goto-char (process-mark process))
      (aidermacs-reset-font-lock-state)
      (insert (propertize command
                          'face 'aidermacs-command-text
                          'font-lock-face 'aidermacs-command-text
                          'rear-nonsticky t))
      (set-marker (process-mark process) (point))
      (comint-send-string process (concat command "\n")))))

(defun aidermacs--send-command-redirect-comint (buffer command)
  "Send command to the aidermacs comint buffer and collect result.
BUFFER is the target buffer.  COMMAND is the text to send.
The output is collected and passed to the current callback."
  (with-current-buffer buffer
    (let ((process (get-buffer-process buffer))
          (output-buffer (get-buffer-create " *aider-redirect-buffer*")))
      (with-current-buffer output-buffer
        (erase-buffer))
      (goto-char (process-mark process))
      (comint-redirect-send-command command output-buffer nil t)
      (unwind-protect
          (while (or quit-flag (null comint-redirect-completed))
            (accept-process-output nil 0.1))
        (unless comint-redirect-completed
          (comint-redirect-cleanup)))
      (aidermacs--store-output (with-current-buffer output-buffer
                                 (buffer-string))))))

(defun aidermacs-comint-interrupt-subjob ()
  "Run `aidermacs--cleanup-temp-buffers' after interrupting a comint subjob."
  (interactive)
  (comint-interrupt-subjob)
  (when (aidermacs--is-aidermacs-buffer-p)
    (aidermacs--cleanup-temp-buffers)))

(defvar aidermacs-comint-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd aidermacs-comint-multiline-newline-key) #'comint-accumulate)
    (define-key map (kbd "C-c C-c") #'aidermacs-comint-interrupt-subjob)
    map)
  "Keymap used when `aidermacs-comint-mode' is enabled.")

(define-minor-mode aidermacs-comint-mode
  "Minor mode for vterm backend buffer used by aidermacs."
  :init-value nil
  :keymap aidermacs-comint-mode-map)

(provide 'aidermacs-backend-comint)

;;; aidermacs-backend-comint.el ends here

;;; aidermacs-output.el --- Output manipulation for Aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.0
;; Keywords: ai emacs llm aider ai-pair-programming tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; SPDX-License-Identifier: Apache-2.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;; This file contains the output and diff functionality for Aidermacs.

;;; Code:

(require 'ediff)
(require 'cl-lib)

(declare-function aidermacs-get-buffer-name "aidermacs")
(declare-function aidermacs-project-root "aidermacs")
(declare-function aidermacs--is-aidermacs-buffer-p "aidermacs")
(declare-function aidermacs--prepare-file-paths-for-command "aidermacs")

(defgroup aidermacs-output nil
  "Output manipulation for Aidermacs."
  :group 'aidermacs)

(defvar-local aidermacs--tracked-files nil
  "List of files that have been mentioned in the aidermacs output.
This is used to avoid having to run /ls repeatedly.")

(defvar-local aidermacs--output-history nil
  "List to store aidermacs output history.
Each entry is a cons cell (timestamp . output-text).")

(defvar-local aidermacs--last-command nil
  "Store the last command sent to aidermacs.")

(defvar-local aidermacs--current-output ""
  "Accumulator for current output being captured as a string.")

(defcustom aidermacs-output-limit 10
  "Maximum number of output entries to keep in history."
  :type 'integer)

(defcustom aidermacs-show-diff-after-change t
  "When non-nil, enable ediff for reviewing AI-generated changes.
When nil, skip preparing temp buffers and showing ediff comparisons."
  :type 'boolean
  :group 'aidermacs)

(defvar-local aidermacs--pre-edit-file-buffers nil
  "Alist of (filename . temp-buffer) storing file state before Aider edits.
These contain the original content of files that might be modified by Aider.")

(defvar-local aidermacs--ediff-queue nil
  "Buffer-local queue of files waiting to be processed by ediff.")

(defvar aidermacs--pre-ediff-window-config nil
  "Window configuration before starting ediff sessions.")

(defun aidermacs--ensure-current-file-tracked ()
  "Ensure current file is tracked in the aidermacs session."
  (when buffer-file-name
    (let* ((session-buffer (get-buffer (aidermacs-get-buffer-name)))
           (filename buffer-file-name)
           (relative-path (file-relative-name filename (aidermacs-project-root))))
      (when session-buffer
        (with-current-buffer session-buffer
          (unless (member relative-path aidermacs--tracked-files)
            (push relative-path aidermacs--tracked-files)
            (let ((command (aidermacs--prepare-file-paths-for-command "/add" (list relative-path))))
              (aidermacs--send-command-backend session-buffer command nil))))))))

(defun aidermacs--capture-file-state (filename)
  "Store the current state of FILENAME in a temporary buffer.
Creates a read-only buffer with the file's content, appropriate major mode,
and syntax highlighting to match the original file."
  (when (and filename (file-exists-p filename))
    (condition-case err
        (let ((temp-buffer (generate-new-buffer
                            (format " *aidermacs-pre-edit:%s*"
                                    (file-name-nondirectory filename)))))
          (with-current-buffer temp-buffer
            (insert-file-contents filename)
            (set-buffer-modified-p nil)
            ;; Use same major mode as the original file
            (let ((buffer-file-name filename)
                  (delay-mode-hooks t))
              (set-auto-mode)
              ;; Ensure syntax highlighting is applied
              (font-lock-mode 1))
            ;; Make buffer read-only
            (setq buffer-read-only t))
          (cons filename temp-buffer))
      (error
       (message "Error capturing file state for %s: %s"
                filename (error-message-string err))
       nil))))

(defun aidermacs--cleanup-temp-buffers ()
  "Clean up all temporary buffers created for ediff sessions.
This is called when all ediff sessions are complete.
Kills all pre-edit buffers that were created to store original file content."
  (interactive)
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    ;; Clean up buffers in the tracking list
    (dolist (file-pair aidermacs--pre-edit-file-buffers)
      (let ((temp-buffer (cdr file-pair)))
        (when (and temp-buffer (buffer-live-p temp-buffer))
          (kill-buffer temp-buffer))))
    ;; Clear the list after cleanup
    (setq aidermacs--pre-edit-file-buffers nil)))

(defun aidermacs--prepare-for-code-edit ()
  "Prepare for code edits by capturing current file states in memory buffers.
Creates temporary buffers containing the original content of all tracked files.
This is skipped if `aidermacs-show-diff-after-change' is nil."
  (when aidermacs-show-diff-after-change
    (when-let* ((files aidermacs--tracked-files))
      (let ((attempts 0)
            (max-attempts 3))
        ;; Use iteration rather than recursion with a limit on attempts
        (while (and (zerop (length aidermacs--pre-edit-file-buffers))
                    (< attempts max-attempts))
          (setq aidermacs--pre-edit-file-buffers
                (cl-remove-duplicates
                 (delq nil
                       (mapcar (lambda (file)
                                 (let* ((clean-file (replace-regexp-in-string " (read-only)$" "" file))
                                        (full-path (expand-file-name clean-file (aidermacs-project-root))))
                                   ;; Only capture state if we don't already have it
                                   (or (assoc full-path aidermacs--pre-edit-file-buffers)
                                       (aidermacs--capture-file-state full-path))))
                               files))
                 :test (lambda (a b) (equal (car a) (car b)))))
          (setq attempts (1+ attempts))
          ;; Add a small delay before retry to allow for file system operations
          (when (and (zerop (length aidermacs--pre-edit-file-buffers))
                     (< attempts max-attempts))
            (sit-for 0.1)))

        (if (zerop (length aidermacs--pre-edit-file-buffers))
            (message "Warning: Failed to capture file states after %d attempts" max-attempts)
          (message "Prepared code edit for %d files" (length aidermacs--pre-edit-file-buffers)))))))

(defun aidermacs--ediff-quit-handler ()
  "Handle ediff session cleanup and process next files in queue.
This function is called when an ediff session is quit and processes
the next file in the ediff queue if any remain."
  (when (and (boundp 'ediff-buffer-A)
             (buffer-live-p ediff-buffer-A)
             (string-match " \\*aidermacs-pre-edit:"
                           (buffer-name ediff-buffer-A)))
    (aidermacs--process-next-ediff-file)))

(defun aidermacs--setup-ediff-cleanup-hooks ()
  "Set up hooks to ensure proper cleanup of temporary buffers after ediff.
Only adds the hook if it's not already present."
  (unless (member #'aidermacs--ediff-quit-handler ediff-quit-hook)
    (add-hook 'ediff-quit-hook #'aidermacs--ediff-quit-handler)))

(defun aidermacs--parse-output-for-files (output)
  "Parse OUTPUT for files and add them to `aidermacs--tracked-files'."
  (when output
    (let ((lines (split-string output "\n"))
          (last-line "")
          (in-udiff nil)
          (current-udiff-file nil))
      (dolist (line lines)
        (cond
         ;; Applied edit to <filename>
         ((string-match "Applied edit to \\(\\./\\)?\\(.+\\)" line)
          (when-let* ((file (match-string 2 line)))
            (add-to-list 'aidermacs--tracked-files file)))

         ;; Added <filename> to the chat.
         ((string-match "Added \\(\\./\\)?\\(.+\\) to the chat" line)
          (when-let* ((file (match-string 2 line)))
            (add-to-list 'aidermacs--tracked-files file)))

         ;; Removed <filename> from the chat (with or without ./ prefix)
         ((string-match "Removed \\(\\./\\)?\\(.+\\) from the chat" line)
          (when-let* ((file (match-string 2 line)))
            (setq aidermacs--tracked-files (delete file aidermacs--tracked-files))))

         ;; Added <filename> to read-only files.
         ((string-match "Added \\(\\./\\)?\\(.+\\) to read-only files" line)
          (when-let* ((file (match-string 2 line)))
            (add-to-list 'aidermacs--tracked-files (concat file " (read-only)"))))

         ;; Moved <file> from editable to read-only files in the chat
         ((string-match "Moved \\(\\./\\)?\\(.+\\) from editable to read-only files in the chat" line)
          (when-let* ((file (match-string 2 line)))
            (let ((editable-file (replace-regexp-in-string " (read-only)$" "" file)))
              (setq aidermacs--tracked-files (delete editable-file aidermacs--tracked-files))
              (add-to-list 'aidermacs--tracked-files (concat file " (read-only)")))))

         ;; Moved <file> from read-only to editable files in the chat
         ((string-match "Moved \\(\\./\\)?\\(.+\\) from read-only to editable files in the chat" line)
          (when-let* ((file (match-string 2 line)))
            (let ((read-only-file (concat file " (read-only)")))
              (setq aidermacs--tracked-files (delete read-only-file aidermacs--tracked-files))
              (add-to-list 'aidermacs--tracked-files file))))

         ;; <file>\nAdd file to the chat?
         ((string-match "Add file to the chat?" line)
          (add-to-list 'aidermacs--tracked-files last-line)
          (aidermacs--prepare-for-code-edit))

         ;; <file> is already in the chat as an editable file
         ((string-match "\\(\\./\\)?\\(.+\\) is already in the chat as an editable file" line)
          (when-let* ((file (match-string 2 line)))
            (add-to-list 'aidermacs--tracked-files file)))

         ;; Handle udiff format
         ;; Detect start of udiff with "--- filename"
         ((string-match "^--- \\(\\./\\)?\\(.+\\)" line)
          (setq in-udiff t
                current-udiff-file (match-string 2 line)))

         ;; Confirm udiff file with "+++ filename" line
         ((and in-udiff
               current-udiff-file
               (string-match "^\\+\\+\\+ \\(\\./\\)?\\(.+\\)" line))
          (let ((plus-file (match-string 2 line)))
            ;; Only add if the filenames match (ignoring ./ prefix)
            (when (string= (file-name-nondirectory current-udiff-file)
                           (file-name-nondirectory plus-file))
              (add-to-list 'aidermacs--tracked-files current-udiff-file)
              (setq in-udiff nil
                    current-udiff-file nil)))))

        (setq last-line line))

      ;; Verify all tracked files exist
      (let* ((project-root (aidermacs-project-root))
             (is-remote (file-remote-p project-root))
             (valid-files nil))
        (dolist (file aidermacs--tracked-files)
          (let* ((is-readonly (string-match-p " (read-only)$" file))
                 (actual-file (if is-readonly
                                  (substring file 0 (- (length file) 12))
                                file))
                 (full-path (expand-file-name actual-file project-root)))
            (when (or (file-exists-p full-path) is-remote)
              (push file valid-files))))
        (setq aidermacs--tracked-files valid-files)))))

(defun aidermacs--store-output (output)
  "Store output string in the history with timestamp.
OUTPUT is the string to store.
If there's a callback function, call it with the output."
  (when (stringp output)
    ;; Store the output
    (setq aidermacs--current-output (substring-no-properties output))
    (push (cons (current-time) (substring-no-properties output)) aidermacs--output-history)
    ;; Trim history if needed
    (when (> (length aidermacs--output-history) aidermacs-output-limit)
      (setq aidermacs--output-history
            (seq-take aidermacs--output-history aidermacs-output-limit)))
    ;; Parse files from output
    (aidermacs--parse-output-for-files output)
    ;; Handle callback if present
    (unless aidermacs--in-callback
      (when (functionp aidermacs--current-callback)
        (let ((aidermacs--in-callback t))
          (funcall aidermacs--current-callback)
          (setq aidermacs--current-callback nil))))))

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
      (local-set-key (kbd "q") #'kill-this-buffer)
      (switch-to-buffer-other-window buf))))

(defun aidermacs-clear-output-history ()
  "Clear the output history."
  (interactive)
  (setq aidermacs--output-history nil))

(defun aidermacs-get-last-output ()
  "Get the most recent output from aidermacs."
  (interactive)
  (when (stringp aidermacs--current-output)
    (message "%s" aidermacs--current-output)
    (kill-new aidermacs--current-output)
    aidermacs--current-output))

(defun aidermacs--detect-edited-files ()
  "Parse current output to find files edited by Aider.
Returns a list of files that have been modified according to the output."
  (let ((project-root (aidermacs-project-root))
        (output aidermacs--current-output)
        (edited-files)
        (unique-files)
        (valid-files))
    (when output
      (with-temp-buffer
        (insert output)
        (goto-char (point-min))

        ;; Case 1: Find "Applied edit to" lines
        (while (search-forward "Applied edit to" nil t)
          (beginning-of-line)
          (when-let* ((file (and (looking-at ".*Applied edit to \\(\\./\\)?\\([^[:space:]]+\\)")
                                (match-string-no-properties 2))))
            (push file edited-files))
          (forward-line 1))

        ;; Case 2: Find triple backtick blocks with filenames
        (goto-char (point-min))
        (while (search-forward "```" nil t)
          (save-excursion
            (forward-line -1)
            (let ((potential-file (string-trim (buffer-substring (line-beginning-position) (line-end-position)))))
              (when (and (not (string-empty-p potential-file))
                         (not (string-match-p "\\`[[:space:]]*\\'" potential-file))
                         (not (string-match-p "^```" potential-file)))
                (push potential-file edited-files))))
          (forward-line 1))

        ;; Case 3: Handle udiff format
        (goto-char (point-min))
        (while (search-forward "--- " nil t)
          (let* ((line-end (line-end-position))
                 (current-udiff-file (buffer-substring (point) line-end)))
            (forward-line 1)
            (when (looking-at "\\+\\+\\+ ")
              (let ((plus-file (buffer-substring (+ (point) 4) (line-end-position))))
                (when (string= (file-name-nondirectory current-udiff-file)
                               (file-name-nondirectory plus-file))
                  (push current-udiff-file edited-files)))))))

      ;; Filter the list to only include valid files
      (setq unique-files (delete-dups edited-files))
      (setq valid-files (nreverse (cl-remove-if-not
                                   (lambda (file)
                                     (file-exists-p (expand-file-name file project-root)))
                                   unique-files)))
      ;; Display a message about which files were changed
      (message "Modified file(s): %s" valid-files)
      valid-files)))

(defun aidermacs--show-ediff-for-edited-files (edited-files)
  "Show ediff for each file in EDITED-FILES.
This is skipped if `aidermacs-show-diff-after-change' is nil."
  (when (and aidermacs-show-diff-after-change edited-files)
    ;; Save current window configuration
    (setq aidermacs--pre-ediff-window-config (current-window-configuration))
    ;; Set up the queue in the current buffer
    (setq-local aidermacs--ediff-queue edited-files)
    ;; Process the first file
    (aidermacs--process-next-ediff-file)))

(defun aidermacs--process-next-ediff-file ()
  "Process the next file in the ediff queue for the current buffer."
  (with-current-buffer (get-buffer (aidermacs-get-buffer-name))
    (if aidermacs--ediff-queue
        (let ((file (pop aidermacs--ediff-queue)))
          (aidermacs--show-ediff-for-file file))
      (aidermacs--cleanup-temp-buffers)
      ;; Restore original window configuration
      (when aidermacs--pre-ediff-window-config
        (set-window-configuration aidermacs--pre-ediff-window-config)
        (setq aidermacs--pre-ediff-window-config nil)))))

(defun aidermacs--show-ediff-for-file (file)
  "Uses the pre-edit buffer stored to compare with the current FILE state."
  (let* ((full-path (expand-file-name file (aidermacs-project-root)))
         (pre-edit-pair (assoc full-path aidermacs--pre-edit-file-buffers))
         (pre-edit-buffer (and pre-edit-pair (cdr pre-edit-pair))))
    (if (and pre-edit-buffer (buffer-live-p pre-edit-buffer))
        (progn
          (let ((current-buffer (or (get-file-buffer full-path)
                                    (find-file-noselect full-path))))
            (with-current-buffer current-buffer
              (revert-buffer t t t))
            (delete-other-windows (get-buffer-window (switch-to-buffer current-buffer)))
            ;; Start ediff session
            (ediff-buffers pre-edit-buffer current-buffer)))
      ;; If no pre-edit buffer found, continue with next file
      (message "No pre-edit buffer found for %s, skipping" file)
      (aidermacs--process-next-ediff-file))))

(provide 'aidermacs-output)
;;; aidermacs-output.el ends here

;;; aidermacs-models.el --- Model selection for aidermacs.el -*- lexical-binding: t; -*-

;;; Commentary:
;; Model selection functionality for aidermacs.el

;;; Code:

(defgroup aidermacs-models nil
  "Model selection customization for aidermacs."
  :group 'aidermacs)

(require 'json)
(require 'url)

(defun fetch-openai-compatible-models (url)
  "Fetch available models from an OpenAI compatible API endpoint at URL."
  (let* ((url-parsed (url-generic-parse-url url))
         (hostname (url-host url-parsed))
         (prefix (cond ((string= hostname "api.openai.com") "openai")
                      ((string= hostname "openrouter.ai") "openrouter")
                      ((string= hostname "api.deepseek.com") "deepseek")
                      ((string= hostname "api.anthropic.com") "anthropic")
                      ((string= hostname "generativelanguage.googleapis.com") "gemini")
                      (t (error "Unknown API host: %s" hostname))))
         (token (cond ((string= hostname "api.openai.com") (getenv "OPENAI_API_KEY"))
                     ((string= hostname "openrouter.ai") (getenv "OPENROUTER_API_KEY"))
                     ((string= hostname "api.deepseek.com") (getenv "DEEPSEEK_API_KEY"))
                     ((string= hostname "api.anthropic.com") (getenv "ANTHROPIC_API_KEY"))
                     ((string= hostname "generativelanguage.googleapis.com") (getenv "GEMINI_API_KEY"))
                     (t (error "Unknown API host: %s" hostname)))))
    (with-current-buffer
        (let ((url-request-extra-headers
               (cond ((string= hostname "api.anthropic.com")
                     `(("x-api-key" . ,token)
                       ("anthropic-version" . "2023-06-01")))
                     ((string= hostname "generativelanguage.googleapis.com")
                      nil)  ; No auth headers for Gemini, key is in URL
                     (t
                      `(("Authorization" . ,(concat "Bearer " token)))))))
          (url-retrieve-synchronously
           (if (string= hostname "generativelanguage.googleapis.com")
               (concat url "/models?key=" token)
             (concat url "/models"))))
      (goto-char url-http-end-of-headers)
      (let* ((json-object-type 'alist)
             (json-data (json-read))
             (models (if (string= hostname "generativelanguage.googleapis.com")
                        (alist-get 'models json-data)
                      (alist-get 'data json-data))))
        (mapcar (lambda (model)
                  (concat prefix "/"
                          (cond
                           ((string= hostname "generativelanguage.googleapis.com")
                            (replace-regexp-in-string "^models/" "" (alist-get 'name model)))
                           ((stringp model) model)  ; Handle case where model is just a string
                           (t (or (alist-get 'id model)
                                (alist-get 'name model))))))
                models)))))

(defun aidermacs--get-supported-models ()
  "Get list of models supported by aider."
  (with-temp-buffer
    (call-process "aider" nil t nil "--list-models" "/")
    (let ((models (seq-filter
                  (lambda (line)
                    (string-prefix-p "- " line))
                  (split-string (buffer-string) "\n" t))))
      (setq models (mapcar (lambda (line)
                            (substring line 2)) ; Remove "- " prefix
                          models))
      (message "Supported models: %S" models)
      models)))

(defun aidermacs--get-available-models ()
  "Get list of available models from multiple providers."
  (let ((models nil)
        (supported-models (aidermacs--get-supported-models)))
    (dolist (url '("https://api.openai.com/v1"
                   "https://openrouter.ai/api/v1"
                   "https://api.deepseek.com"
                   "https://api.anthropic.com/v1"
                   "https://generativelanguage.googleapis.com/v1beta"))
      (condition-case err
          (let* ((fetched-models (fetch-openai-compatible-models url))
                 (filtered-models (seq-filter (lambda (model)
                                             (member model supported-models))
                                           fetched-models)))
            (message "Fetched models from %s: %S" url fetched-models)
            (message "Filtered models from %s: %S" url filtered-models)
            (setq models (append models filtered-models)))
        (error (message "Failed to fetch models from %s: %s" url err))))
    models))

(defun aidermacs--select-model ()
  "Private function for model selection with completion."
  (let ((models (aidermacs--get-available-models)))
    (completing-read "Select AI model: " models nil t)))

;;;###autoload
(defun aidermacs-change-model ()
  "Interactively select and change AI model in current aidermacs session."
  (interactive)
  (let ((model (aidermacs--select-model)))
    (when model
      (aidermacs--send-command (format "/model %s" model) t))))

(provide 'aidermacs-models)

;;; aidermacs-models.el ends here

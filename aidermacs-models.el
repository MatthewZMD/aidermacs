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
                      (t (error "Unknown API host: %s" hostname))))
         (token (cond ((string= hostname "api.openai.com") (getenv "OPENAI_API_KEY"))
                     ((string= hostname "openrouter.ai") (getenv "OPENROUTER_API_KEY"))
                     ((string= hostname "api.deepseek.com") (getenv "DEEPSEEK_API_KEY"))
                     ((string= hostname "api.anthropic.com") (getenv "ANTHROPIC_API_KEY"))
                     (t (error "Unknown API host: %s" hostname)))))
    (with-current-buffer
        (let ((url-request-extra-headers
               (if (string= hostname "api.anthropic.com")
                   `(("x-api-key" . ,token)
                     ("anthropic-version" . "2023-06-01"))
                 `(("Authorization" . ,(concat "Bearer " token))))))
          (url-retrieve-synchronously (concat url "/models")))
      (goto-char url-http-end-of-headers)
      (let* ((json-object-type 'alist)
             (json-data (json-read))
             (models (alist-get 'data json-data)))
        (mapcar (lambda (model)
                  (concat prefix "/" (alist-get 'id model)))
                models)))))

(defun aidermacs--get-available-models ()
  "Get list of available models from multiple providers."
  (let ((models nil))
    (dolist (url '("https://api.openai.com/v1"
                   "https://openrouter.ai/api/v1"
                   "https://api.deepseek.com"
                   "https://api.anthropic.com/v1"))
      (condition-case err
          (setq models (append models (fetch-openai-compatible-models url)))
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

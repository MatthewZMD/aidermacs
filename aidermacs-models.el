;;; aidermacs-models.el --- Model selection for aidermacs.el -*- lexical-binding: t; -*-
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
;; This file provides model selection functionality for Aidermacs,
;; allowing users to choose between different AI models for their
;; Aider sessions. It supports fetching models from various API
;; providers and caching them for faster access.
;;
;; Key features include:
;; - Selection of AI models via a completing-read interface.
;; - Fetching of available models from OpenAI-compatible API endpoints.
;; - Caching of models for faster access.
;; - Customization of default models for different tasks.
;;
;;; Code:

(require 'json)
(require 'url)

(declare-function aidermacs--send-command "aidermacs" (command &optional no-switch-to-buffer use-existing redirect callback))
(declare-function aidermacs-buffer-name "aidermacs" ())
(declare-function aidermacs-exit "aidermacs" ())

(defgroup aidermacs-models nil
  "Model selection customization for aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-default-model "sonnet"
  "Default AI model to use for aidermacs sessions when not in Architect mode."
  :type 'string
  :group 'aidermacs-models)

(defcustom aidermacs-architect-model "sonnet"
  "Default AI model to use for architectural reasoning in aidermacs sessions."
  :type 'string
  :group 'aidermacs-models)

(defcustom aidermacs-editor-model aidermacs-default-model
  "Default AI model to use for code editing in aidermacs sessions.
Defaults to `aidermacs-default-model` if not explicitly set."
  :type 'string
  :group 'aidermacs-models)

(defcustom aidermacs-use-architect-mode nil
  "If non-nil, use separate Architect/Editor mode."
  :type 'boolean
  :group 'aidermacs-models)

(defcustom aidermacs-popular-models
  '("sonnet"
    "o1-mini"
    "gemini/gemini-2.0-flash"
    "r1"
    "deepseek/deepseek-chat")
  "List of available AI models for selection.
Each model should be in the format expected by the aidermacs CLI.
Also based on aidermacs LLM benchmark: https://aidermacs.chat/docs/leaderboards/"
  :type '(repeat string)
  :group 'aidermacs-models)

(defvar aidermacs--cached-models aidermacs-popular-models
  "Cache of available AI models.")

(defun aidermacs--fetch-openai-compatible-models (url)
  "Fetch available models from an OpenAI compatible API endpoint.
URL should be the base API endpoint, e.g. https://api.openai.com/v1.
Returns a list of model names with appropriate prefixes based on the
API provider."
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
    (with-local-quit
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
                  models))))))


(defun aidermacs--select-model ()
  "Provide model selection with completion.
This is a private function used internally."
  (condition-case nil
      (let ((model (completing-read "Select AI model: " aidermacs--cached-models nil t)))
        (when model
          (aidermacs--send-command (format "/model %s" model))))
    (quit (message "Model selection cancelled"))))

(defun aidermacs--get-available-models ()
  "Get list of models supported by aider using the /models command.
This fetches models from various API providers and caches them."
  (aidermacs--send-command
   "/models /" nil nil t
   (lambda ()
     (let* ((supported-models
             (seq-filter
              (lambda (line)
                (string-prefix-p "- " line))
              (split-string aidermacs--current-output "\n" t)))
            (models nil))
       (setq supported-models
             (mapcar (lambda (line)
                       (substring line 2)) ; Remove "- " prefix
                     supported-models))
       (dolist (url-token-pair '(("https://api.openai.com/v1" . "OPENAI_API_KEY")
                                 ("https://openrouter.ai/api/v1" . "OPENROUTER_API_KEY")
                                 ("https://api.deepseek.com" . "DEEPSEEK_API_KEY")
                                 ("https://api.anthropic.com/v1" . "ANTHROPIC_API_KEY")
                                 ("https://generativelanguage.googleapis.com/v1beta" . "GEMINI_API_KEY")))
         (let ((url (car url-token-pair))
               (token-value (getenv (cdr url-token-pair))))
           (when (and token-value (not (string-empty-p token-value)))
             (condition-case err
                 (let* ((fetched-models (aidermacs--fetch-openai-compatible-models url))
                        (filtered-models (seq-filter (lambda (model)
                                                       (member model supported-models))
                                                     fetched-models)))
                   (setq models (append models filtered-models)))
               (error (message "Failed to fetch models from %s: %s" url (error-message-string err)))))))
       (setq aidermacs--cached-models models)
       (aidermacs--select-model)))))

(defun aidermacs-clear-model-cache ()
  "Clear the cached models, forcing a fresh fetch on next use.
This is useful when available models have changed."
  (interactive)
  (setq aidermacs--cached-models nil)
  (message "Model cache cleared"))

;;;###autoload
(defun aidermacs-change-model ()
  "Interactively select and change AI model in current aidermacs session."
  (interactive)
  (when (and aidermacs--cached-models
             (equal aidermacs--cached-models aidermacs-popular-models)
             (fboundp 'aidermacs-get-buffer-name)
             (get-buffer (aidermacs-get-buffer-name)))
    (setq aidermacs--cached-models nil))

  (if aidermacs--cached-models
      (aidermacs--select-model)
    (aidermacs--get-available-models)))

(provide 'aidermacs-models)
;;; aidermacs-models.el ends here

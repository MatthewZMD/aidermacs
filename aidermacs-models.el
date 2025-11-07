;;; aidermacs-models.el --- Model selection for aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.5
;; Keywords: ai emacs llm aider ai-pair-programming tools
;; URL: https://github.com/MatthewZMD/aidermacs
;; SPDX-License-Identifier: Apache-2.0

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Provides model selection for Aidermacs, allowing choice between
;; different AI models for Aider sessions. Supports fetching models
;; from various API providers and caching for faster access.
;;
;; Features:
;; - Model selection via completing-read interface
;; - Fetch models from OpenAI-compatible APIs
;; - Model caching for faster access
;; - Custom default models for different tasks

;; Originally forked from: Kang Tu <tninja@gmail.com> Aider.el

;;; Code:

(require 'json)
(require 'url)

(declare-function aidermacs--send-command "aidermacs")
(declare-function aidermacs-buffer-name "aidermacs")
(declare-function aidermacs-exit "aidermacs")
(declare-function aidermacs-aider-version "aidermacs")
(declare-function aidermacs-get-buffer-name "aidermacs")

(defvar aidermacs--current-output)
(defvar aidermacs-use-architect-mode)
(defvar aidermacs--current-mode)

(defvar url-http-end-of-headers)

(defgroup aidermacs-models nil
  "Model selection for Aidermacs."
  :group 'aidermacs)

(defcustom aidermacs-default-model (or (getenv "AIDER_MODEL") "sonnet")
  "Default AI model to use for aidermacs sessions when not in Architect mode.
Respects the `AIDER_MODEL' environment variable if set."
  :type 'string)

(defcustom aidermacs-architect-model nil
  "Default reasoning AI model to use for architect mode.
If nil, uses the value of `aidermacs-default-model'."
  :type '(choice (const :tag "Use default model" nil)
                 (string :tag "Specific model")))

(defcustom aidermacs-editor-model (getenv "AIDER_EDITOR_MODEL")
  "Default editing AI model to use for architect mode.
If nil, uses the value of `aidermacs-default-model'.
Respects the `AIDER_EDITOR_MODEL' environment variable if set."
  :type '(choice (const :tag "Use default model" nil)
                 (string :tag "Specific model")))

(defcustom aidermacs-weak-model (getenv "AIDER_WEAK_MODEL")
  "Default weak AI model to use.
If nil, uses a model automatically selected based on the default model.
Respects the `AIDER_WEAK_MODEL' environment variable if set."
  :type '(choice (const :tag "Use default model" nil)
                 (string :tag "Specific model")))

(defvar aidermacs--cached-models nil
  "Cache of available AI models.")

(defun aidermacs-get-architect-model ()
  "Get the effective architect model, falling back to default if not set."
  (or aidermacs-architect-model aidermacs-default-model))

(defun aidermacs-get-editor-model ()
  "Get the effective editor model, falling back to default if not set."
  (or aidermacs-editor-model aidermacs-default-model))

(defun aidermacs-get-weak-model ()
  "Get the effective weak model, falling back to default if not set."
  (or aidermacs-weak-model aidermacs-default-model))

(defconst aidermacs--api-providers
  '(("https://openrouter.ai/api/v1" . ((hostname . "openrouter.ai")
                                       (prefix . "openrouter")
                                       (env-var . "OPENROUTER_API_KEY")
                                       (auth-header . (("Authorization" . "Bearer %s")))
                                       (models-path . "/models")
                                       (models-key . data)))
    ("https://api.openai.com/v1" . ((hostname . "api.openai.com")
                                    (prefix . "openai")
                                    (env-var . "OPENAI_API_KEY")
                                    (auth-header . (("Authorization" . "Bearer %s")))
                                    (models-path . "/models")
                                    (models-key . data)))
    ("https://api.deepseek.com" . ((hostname . "api.deepseek.com")
                                   (prefix . "deepseek")
                                   (env-var . "DEEPSEEK_API_KEY")
                                   (auth-header . (("Authorization" . "Bearer %s")))
                                   (models-path . "/models")
                                   (models-key . data)))
    ("https://api.anthropic.com/v1" . ((hostname . "api.anthropic.com")
                                       (prefix . "anthropic")
                                       (env-var . "ANTHROPIC_API_KEY")
                                       (auth-header . (("x-api-key" . "%s")
                                                       ("anthropic-version" . "2023-06-01")))
                                       (models-path . "/models")
                                       (models-key . data)))
    ("https://generativelanguage.googleapis.com/v1beta" . ((hostname . "generativelanguage.googleapis.com")
                                                           (prefix . "gemini")
                                                           (env-var . "GEMINI_API_KEY")
                                                           (auth-header . nil)
                                                           (models-path . "/models?key=%s")
                                                           (models-key . models)
                                                           (model-name-transform . (lambda (name)
                                                                                     (replace-regexp-in-string "^models/" "" name)))))
    ("https://api.x.ai/v1" . ((hostname . "api.x.ai")
                              (prefix . "xai")
                              (env-var . "XAI_API_KEY")
                              (auth-header . (("Authorization" . "Bearer %s")))
                              (models-path . "/models")
                              (models-key . data))))
  "Configuration for different API providers.
Each entry maps a base URL to a configuration alist with:
- hostname: The API hostname
- prefix: Prefix to add to model names
- env-var: Environment variable containing the API key
- auth-header: Headers for authentication (nil if not needed)
- models-path: Path to fetch models, with %s for token if needed
- models-key: JSON key containing the models list
- model-name-transform: Optional function to transform model names")

(defun aidermacs--format-price (pricing)
  "Format pricing information into a string.
PRICING is an alist that may contain 'prompt' and 'completion' prices."
  (when pricing
    (let* ((prompt-price-str (alist-get 'prompt pricing "0"))
           (completion-price-str (alist-get 'completion pricing "0"))
           (prompt-price (ignore-errors (string-to-number prompt-price-str)))
           (completion-price (ignore-errors (string-to-number completion-price-str))))
      (if (and prompt-price completion-price (> (+ prompt-price completion-price) 0))
          (format "($%.2f/$%.2f/M)"
                  (* prompt-price 1000000)
                  (* completion-price 1000000))
        ""))))

(defun aidermacs--fetch-openai-compatible-models (url token)
  "Fetch available models from an OpenAI compatible API endpoint.
URL should be the base API endpoint, e.g. https://api.openai.com/v1.
TOKEN is the API token for authentication.
Returns a list of model names with appropriate prefixes based on the
API provider."
  (let* ((provider-config (cdr (assoc url aidermacs--api-providers)))
         (prefix (alist-get 'prefix provider-config))
         (auth-headers (alist-get 'auth-header provider-config))
         (models-path (alist-get 'models-path provider-config))
         (models-key (alist-get 'models-key provider-config))
         (transform-fn (alist-get 'model-name-transform provider-config)))

    (unless provider-config
      (error "Unknown API URL: %s" url))

    (with-local-quit
      (with-current-buffer
          (let ((url-request-extra-headers
                 (when auth-headers
                   (mapcar (lambda (header)
                             (cons (car header)
                                   (format (cdr header) token)))
                           auth-headers))))
            (url-retrieve-synchronously
             (concat url (if (string-match-p "%s" models-path)
                             (format models-path token)
                           models-path))))

        (goto-char url-http-end-of-headers)
        (let* ((json-object-type 'alist)
               (json-data (json-read))
               (models (alist-get models-key json-data)))
          (mapcar (lambda (model)
                    (let* ((model-id
                            (cond
                             ((stringp model) model)
                             (transform-fn (funcall transform-fn (alist-get 'name model)))
                             (t (or (alist-get 'id model) (alist-get 'name model)))))
                           (full-model-id (concat prefix "/" model-id))
                           (pricing (unless (stringp model) (alist-get 'pricing model)))
                           (price-str (aidermacs--format-price pricing)))
                      `((id . ,full-model-id) (price-str . ,price-str))))
                  models))))))

(defun aidermacs--select-model (&optional set-weak-model)
  "Provide model selection with completion, handling main/weak/editor models.
When SET-WEAK-MODEL is non-nil, only allow setting the weak model."
  (condition-case nil
      (let* ((aider-version (aidermacs-aider-version))
             (supports-specific-model (version<= "0.78.0" aider-version))
             (is-architect-mode (and (eq aidermacs--current-mode 'architect) supports-specific-model))
             (set-weak-model (and set-weak-model supports-specific-model))
             (model-type
              (cond
               (set-weak-model "Weak Model")
               (is-architect-mode
                (completing-read
                 "Select model type: "
                 '("Main/Reasoning Model" "Editing Model")
                 nil nil))
               (t "Main Model")))
             (candidates (mapcar (lambda (m)
                                   (let ((id (alist-get 'id m))
                                         (price (alist-get 'price-str m)))
                                     (cons (if (string-empty-p price)
                                               id
                                             (format "%-60s %s" id price))
                                           id)))
                                 aidermacs--cached-models))
             (model (completing-read (format "Select %s: " model-type) candidates nil nil)))
        (when model
          (cond
           (set-weak-model
            (setq aidermacs-weak-model model)
            (aidermacs--send-command (format "/weak-model %s" model)))
           ((and is-architect-mode supports-specific-model)
            (pcase model-type
              ("Main/Reasoning Model"
               (setq aidermacs-architect-model model)
               (aidermacs--send-command (format "/model %s" model)))
              ("Editing Model"
               (setq aidermacs-editor-model model)
               (aidermacs--send-command (format "/editor-model %s" model)))))
           (t
            (setq aidermacs-default-model model)
            (aidermacs--send-command (format "/model %s" model))))))
    (quit (message "Model selection cancelled"))))

(defun aidermacs--get-available-models ()
  "Get list of models supported by aider using the /models command."
  (aidermacs--send-command
   "/models /" nil nil t
   (lambda ()
     (let* ((all-models
             (mapcar (lambda (line)
                       (substring line 2)) ; Remove "- " prefix
                     (seq-filter
                      (lambda (line)
                        (string-prefix-p "- " line))
                      (split-string aidermacs--current-output "\n" t))))
            (models))
       (dolist (provider-entry aidermacs--api-providers)
         (let* ((url (car provider-entry))
                (config (cdr provider-entry))
                (env-var (alist-get 'env-var config))
                (token-value (getenv env-var)))
           (when (and token-value (not (string-empty-p token-value)))
             (condition-case err
                 (let* ((fetched-models (aidermacs--fetch-openai-compatible-models url token-value))
                        (filtered-models (seq-filter (lambda (model)
                                                       (member (alist-get 'id model) all-models))
                                                     fetched-models)))
                   (setq models (append models filtered-models)))
               (error "Failed to fetch models from %s: %s" url (error-message-string err))))))
       ;; If we couldn't fetch any models from APIs, just use all supported models list
       (if models
           (setq aidermacs--cached-models models)
         (setq aidermacs--cached-models
               (mapcar (lambda (m) `((id . ,m) (price-str . ""))) all-models)))))))

(defun aidermacs-clear-model-cache ()
  "Clear the cached models, forcing a fresh fetch on next use.
This is useful when available models have changed."
  (interactive)
  (setq aidermacs--cached-models nil)
  (message "Model cache cleared"))

(defun aidermacs-change-model (&optional arg)
  "Interactively select and change AI model in current aidermacs session.
With prefix ARG, only allow setting the weak model."
  (interactive "P")
  (unless aidermacs--cached-models
    (aidermacs--get-available-models))
  (aidermacs--select-model arg))

(provide 'aidermacs-models)
;;; aidermacs-models.el ends here

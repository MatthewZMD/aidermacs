;;; aidermacs-models.el --- Model selection for aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.7
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

(defcustom aidermacs-litellm-prices-file nil
  "Manual path to litellm model_prices_and_context_window.json.
If set, use this path directly instead of searching.
Example: \"~/.local/lib/python3.11/site-packages/litellm/model_prices_and_context_window.json\""
  :type '(choice (const :tag "Auto-detect" nil)
                 (file :tag "Specify path"))
  :group 'aidermacs-models)

(defcustom aidermacs-litellm-prices-cache-duration 86400
  "Duration in seconds to cache litellm prices (default: 1 day)."
  :type 'integer
  :group 'aidermacs-models)

(defvar aidermacs--litellm-prices-cache nil
  "Cache of litellm model prices. Alist mapping model-id to ((input-price . val) (output-price . val)).")

(defvar aidermacs--litellm-prices-cache-timestamp nil
  "Timestamp when litellm prices were last fetched.")

(defvar aidermacs--litellm-file-path-cache nil
  "Cache of the litellm prices file path.")

(defun aidermacs--find-litellm-prices-file ()
  "Find the local litellm prices file from Aider's installation."
  (or aidermacs--litellm-file-path-cache
      (when aidermacs-litellm-prices-file
        (let ((expanded (expand-file-name aidermacs-litellm-prices-file)))
          (when (file-exists-p expanded)
            (message "Using user-specified litellm prices file: %s" expanded)
            (setq aidermacs--litellm-file-path-cache expanded))))
      (let ((possible-patterns
             (append
              ;; Aider-specific paths
              '("~/.aider/caches/model_prices_and_context_window.json"
                "~/.aider/caches/litellm/model_prices_and_context_window.json"
                "~/.aider/lib/python*/site-packages/litellm/model_prices_and_context_window.json")
              ;; User-local and system-wide pip
              '("~/.local/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "/usr/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "/usr/local/lib/python*/site-packages/litellm/model_prices_and_context_window.json")
              ;; macOS Homebrew
              '("/opt/homebrew/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "/usr/local/opt/python*/libexec/lib/python*/site-packages/litellm/model_prices_and_context_window.json")
              ;; Pip cache
              '("~/.cache/pip/wheels/*/litellm-*/litellm/model_prices_and_context_window.json"
                "~/.cache/pip/pool/*/litellm-*/litellm/model_prices_and_context_window.json"
                "~/.cache/pip/*/litellm*/model_prices_and_context_window.json")
              ;; Python version managers
              '("~/.pyenv/versions/*/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "~/.asdf/installs/python/*/lib/python*/site-packages/litellm/model_prices_and_context_window.json")
              ;; Conda/Anaconda
              '("~/anaconda3/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "~/miniconda3/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "~/.conda/envs/*/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "~/mambaforge/lib/python*/site-packages/litellm/model_prices_and_context_window.json")
              ;; macOS user Python
              '("~/Library/Python/*/lib/python*/site-packages/litellm/model_prices_and_context_window.json")
              ;; Virtual environments
              '("~/.virtualenvs/*/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "~/venv/*/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "*/venv/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "*/.venv/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                ".venv/lib/python*/site-packages/litellm/model_prices_and_context_window.json"
                "venv/lib/python*/site-packages/litellm/model_prices_and_context_window.json"))))
        (setq aidermacs--litellm-file-path-cache
              (cl-some (lambda (pattern)
                         (let ((matches (file-expand-wildcards pattern t)))
                           (when matches
                             (car matches))))
                       possible-patterns))
        (unless aidermacs--litellm-file-path-cache
          (message "Could not find litellm prices file. Set `aidermacs-litellm-prices-file' manually"))
        aidermacs--litellm-file-path-cache)))

(defun aidermacs--read-litellm-prices ()
  "Read model prices from local litellm JSON file."
  (condition-case err
      (let ((file-path (aidermacs--find-litellm-prices-file)))
        (when file-path
          (with-temp-buffer
            (insert-file-contents file-path)
            (let ((json-object-type 'alist)
                  (json-data (json-read)))
              (delq nil
                    (mapcar (lambda (entry)
                              (when (consp entry)
                                (let* ((model-id (format "%s" (car entry)))
                                       (info (cdr entry))
                                       ;; Handle both symbol and string keys
                                       (input-price (or (alist-get 'input_cost_per_token info)
                                                       (alist-get "input_cost_per_token" info)))
                                       (output-price (or (alist-get 'output_cost_per_token info)
                                                        (alist-get "output_cost_per_token" info))))
                                  ;; Only keep entries with pricing information
                                  (when (and model-id (or input-price output-price))
                                    (cons model-id
                                          `((input-price . ,input-price)
                                            (output-price . ,output-price)))))))
                            (cl-remove-if (lambda (entry)
                                           (and (consp entry)
                                                (member (car entry) '(sample_spec "sample_spec"))))
                                          json-data)))))))
    (error
     (message "Failed to read litellm prices: %s" (error-message-string err))
     nil)))

(defun aidermacs--get-litellm-prices ()
  "Get litellm prices, using cache if still valid."
  (if (and aidermacs--litellm-prices-cache
           aidermacs--litellm-prices-cache-timestamp
           (< (- (float-time) aidermacs--litellm-prices-cache-timestamp)
              aidermacs-litellm-prices-cache-duration))
      aidermacs--litellm-prices-cache
    (let ((prices (aidermacs--read-litellm-prices)))
      (when prices
        (setq aidermacs--litellm-prices-cache prices)
        (setq aidermacs--litellm-prices-cache-timestamp (float-time)))
      prices)))

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


(defun aidermacs--model-total-price (model)
  "Calculate total price for MODEL from pricing info.
Returns a number, or 999999 if price cannot be determined."
  (let* ((price-str (alist-get 'price-str model)))
    (if (and price-str (string-match "($\\([0-9.]+\\)/$\\([0-9.]+\\)/M)" price-str))
        (+ (string-to-number (match-string 1 price-str))
           (string-to-number (match-string 2 price-str)))
      999999)))

(defun aidermacs--get-cheapest-models (models count)
  "Return the cheapest COUNT models from MODELS.
Returns a list of (model . rank) cons cells, where rank starts from 1."
  (let* ((models-with-price (mapcar (lambda (m) (cons m (aidermacs--model-total-price m))) models))
         (sorted (sort (copy-sequence models-with-price) (lambda (a b) (< (cdr a) (cdr b)))))
         (top-n (seq-take sorted count)))
    (cl-loop for idx from 1 to (min count (length top-n))
             for item in top-n
             collect (cons (car item) idx))))

(defun aidermacs--make-model-annotator (cheapest-models)
  "Create annotation function for the cheapest models.
CHEAPEST-MODELS is a list of (model . rank) from `aidermacs--get-cheapest-models'."
  (let ((rank-map (make-hash-table :test 'equal))
        (ids (mapcar (lambda (pair) (alist-get 'id (car pair))) cheapest-models)))
    (dolist (entry cheapest-models)
      (puthash (alist-get 'id (car entry)) (cdr entry) rank-map))
    (lambda (cand-id)
      (when-let ((rank (gethash cand-id rank-map)))
        (format " [Rank %d - Cheapest]" rank)))))

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
             (annotator (aidermacs--make-model-annotator (aidermacs--get-cheapest-models aidermacs--cached-models 10)))
             (candidates
              (mapcar (lambda (m)
                        (let* ((id (alist-get 'id m))
                               (id-str (if (stringp id) id (format "%s" id)))
                               (price-str (alist-get 'price-str m))
                               (price-str-safe (if (stringp price-str) price-str ""))
                               (display-str (if (string-empty-p price-str-safe)
                                                id-str
                                              (format "%-80s %s" id-str price-str-safe))))
                          (cons display-str id-str)))
                      aidermacs--cached-models))
             (model (completing-read
                     (format "Select %s: " model-type)
                     (lambda (str pred action)
                       (if (eq action 'metadata)
                           `(metadata
                             (annotation-function . ,(lambda (cand) (funcall annotator (cdr (assoc cand candidates)))))
                             (display-sort-function . identity)
                             (cycle-sort-function . identity))
                         (complete-with-action action candidates str pred)))
                     nil t)))
        (when model
          (let ((real-model (cdr (assoc model candidates))))
            (when real-model
              (cond
               (set-weak-model
                (setq aidermacs-weak-model real-model)
                (aidermacs--send-command (format "/weak-model %s" real-model)))
               ((and is-architect-mode supports-specific-model)
                (pcase model-type
                  ("Main/Reasoning Model"
                   (setq aidermacs-architect-model real-model)
                   (aidermacs--send-command (format "/model %s" real-model)))
                  ("Editing Model"
                   (setq aidermacs-editor-model real-model)
                   (aidermacs--send-command (format "/editor-model %s" real-model)))))
               (t
                (setq aidermacs-default-model real-model)
                (aidermacs--send-command (format "/model %s" real-model))))))))
    (quit (message "Model selection cancelled"))))

(defun aidermacs--parse-model-identity (model-id)
  "Parse MODEL-ID into canonical identity components.
Returns an alist with keys: provider, family, variant, full-id.
Examples:
  \"openai/gpt-4o-2024-08-06\" -> ((provider . \"openai\") (family . \"gpt-4o\") ...)
  \"claude-3-5-sonnet-20241022\" -> ((provider . nil) (family . \"claude-3-5-sonnet\") ...)"
  (unless (stringp model-id)
    (message "Warning: model-id is not a string: %S (type: %s)" model-id (type-of model-id))
    (setq model-id (format "%s" model-id)))
  (let* ((parts (split-string model-id "/"))
         (has-provider (> (length parts) 1))
         (provider (if has-provider (car parts) nil))
         (base (if has-provider (mapconcat #'identity (cdr parts) "/") model-id))
         ;; Extract variant (date or version suffix)
         (variant (when (string-match "-\\([0-9]\\{4\\}-[0-9]\\{2\\}-[0-9]\\{2\\}\\|-[0-9]\\{6,8\\}\\|-latest\\)$" base)
                    (match-string 1 base)))
         (family (if variant
                     (substring base 0 (- (length base) (length variant)))
                   base)))
    `((provider . ,provider)
      (family . ,family)
      (variant . ,variant)
      (full-id . ,model-id))))

(defun aidermacs--match-model-price (model-id litellm-prices)
  "Find price info for MODEL-ID from LITELLM-PRICES using cascade matching."
  (unless (stringp model-id)
    (message "Warning: aidermacs--match-model-price received non-string model-id: %S" model-id)
    (setq model-id (format "%s" model-id)))
  (when litellm-prices
    (let ((identity (aidermacs--parse-model-identity model-id))
          (result nil))
      ;; Strategy 1: Exact match
      (setq result (cdr (assoc model-id litellm-prices)))

      ;; Strategy 2: Family match (strip provider)
      (unless result
        (let ((family (alist-get 'family identity)))
          (when family
            (setq result (cdr (assoc family litellm-prices))))))

      ;; Strategy 3: Fuzzy family match
      (unless result
        (let ((target-family (alist-get 'family identity)))
          (dolist (entry litellm-prices)
            (when (not result)
              (let ((entry-key (car entry)))
                (when (stringp entry-key)
                  (let ((entry-identity (aidermacs--parse-model-identity entry-key)))
                    (when (and (string= target-family (alist-get 'family entry-identity))
                               (or (null (alist-get 'provider identity))
                                   (null (alist-get 'provider entry-identity))
                                   (string= (alist-get 'provider identity)
                                            (alist-get 'provider entry-identity))))
                      (setq result (cdr entry))))))))))

      ;; Strategy 4: Provider/family combination
      (unless result
        (let ((provider (alist-get 'provider identity))
              (family (alist-get 'family identity)))
          (when (and provider family)
            (setq result (cdr (assoc (concat provider "/" family) litellm-prices))))))

      ;; Strategy 5: Substring match
      (unless result
        (let ((family (alist-get 'family identity)))
          (when family
            (setq result (cdr (cl-find-if (lambda (entry)
                                            (let ((entry-key (car entry)))
                                              (and (stringp entry-key)
                                                   (string-match-p (regexp-quote family) entry-key))))
                                          litellm-prices))))))

      result)))



(defun aidermacs--get-available-models (&optional callback)
  "Get list of models supported by aider using the /models command.
Prices are fetched from local litellm JSON file with cascade matching.
If API keys are configured, only show models from those providers.
CALLBACK is called after models are fetched and cached."
  (aidermacs--send-command
   "/models /" nil nil t
   (lambda ()
     (if (not (stringp aidermacs--current-output))
         (progn
           (message "Error: aidermacs--current-output is not a string: %S (type: %s)"
                    aidermacs--current-output (type-of aidermacs--current-output))
           (setq aidermacs--cached-models nil)
           (when callback (funcall callback)))
       (let* ((all-models
               (mapcar (lambda (line) (substring line 2))
                       (seq-filter (lambda (line) (string-prefix-p "- " line))
                                   (split-string aidermacs--current-output "\n" t))))
              (all-models-str (mapcar (lambda (m) (if (stringp m) m (format "%s" m))) all-models))
              (litellm-prices (aidermacs--get-litellm-prices))
              (models))
         (dolist (model-id all-models-str)
           (when (stringp model-id)
             (let* ((price-info (aidermacs--match-model-price model-id litellm-prices))
                    (price-str (if price-info
                                   (let ((input-price (alist-get 'input-price price-info))
                                         (output-price (alist-get 'output-price price-info)))
                                     (if (and input-price output-price
                                              (numberp input-price) (numberp output-price)
                                              (> (+ input-price output-price) 0))
                                         (format "($%.2f/$%.2f/M)"
                                                 (* input-price 1000000)
                                                 (* output-price 1000000))
                                       ""))
                                 "")))
               (push `((id . ,model-id) (price-str . ,price-str)) models))))

         (let ((final-models (or (nreverse models)
                               (mapcar (lambda (m)
                                         (if (stringp m)
                                             `((id . ,m) (price-str . ""))
                                           `((id . ,(format "%s" m)) (price-str . ""))))
                                       all-models-str))))
           (setq aidermacs--cached-models final-models)
           (when callback (funcall callback))))))))

(defun aidermacs-clear-model-cache ()
  "Clear the cached models and litellm prices, forcing a fresh fetch on next use."
  (interactive)
  (setq aidermacs--cached-models nil)
  (setq aidermacs--litellm-prices-cache nil)
  (setq aidermacs--litellm-prices-cache-timestamp nil)
  (setq aidermacs--litellm-file-path-cache nil)
  (message "Model cache cleared"))

(defun aidermacs-change-model (&optional arg)
  "Interactively select and change AI model in current aidermacs session.
With prefix ARG, only allow setting the weak model."
  (interactive "P")
  (if aidermacs--cached-models
      (aidermacs--select-model arg)
    (message "Fetching available models...")
    (aidermacs--get-available-models
     (lambda ()
       (message "Models fetched successfully")
       (aidermacs--select-model arg)))))

(provide 'aidermacs-models)
;;; aidermacs-models.el ends here

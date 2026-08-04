;;; aidermacs-models.el --- Model selection for aidermacs -*- lexical-binding: t; -*-
;; Author: Mingde (Matthew) Zeng <matthewzmd@posteo.net>
;; Version: 1.11
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
(declare-function aidermacs-project-root "aidermacs")

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
Example: \"/path/to/litellm/model_prices_and_context_window.json\""
  :type '(choice (const :tag "Auto-detect" nil)
                 (file :tag "Specify path"))
  :group 'aidermacs-models)

(defcustom aidermacs-litellm-prices-cache-duration 86400
  "Duration in seconds to cache litellm prices (default: 1 day)."
  :type 'integer
  :group 'aidermacs-models)

(defcustom aidermacs-model-filter-mode 'all
  "How to filter models in the selection list.
- `configured-only': Only show models from the model settings file
- `configured-first': Show configured models first, then all others
- `all': Show all models (no filtering)"
  :type '(choice (const :tag "Only configured models" configured-only)
                 (const :tag "Configured models first" configured-first)
                 (const :tag "All models" all))
  :group 'aidermacs-models)

(defvar aidermacs--litellm-prices-cache nil
  "Cache of litellm model prices.
Alist mapping model-id to ((input-price . val) (output-price . val)).")

(defvar aidermacs--litellm-prices-cache-timestamp nil
  "Timestamp when litellm prices were last fetched.")

(defvar aidermacs--litellm-file-path-cache nil
  "Cache of the litellm prices file path.")

(defun aidermacs--find-model-settings-file ()
  "Find the aider model settings YAML file.
Searches in order: env var, homedir, git root, cwd.
Default filename is `.aider.model.settings.yml' (matching aider's behavior)."
  (or (getenv "AIDER_MODEL_SETTINGS_FILE")
      (let* ((default-file ".aider.model.settings.yml")
             (patterns
              (list
               (expand-file-name (concat "~/" default-file))  ; homedir
               (when-let ((root (aidermacs-project-root)))
                 (expand-file-name default-file root))        ; git root
               (expand-file-name default-file)                ; cwd
               )))
        (cl-some (lambda (p)
                   (when (and p (file-exists-p p)) p))
                 patterns))))

(defun aidermacs--read-configured-models ()
  "Read model names from the aider model settings YAML file.
Returns a list of model name strings extracted from '- name:' entries."
  (let ((file (aidermacs--find-model-settings-file)))
    (when (and file (file-exists-p file))
      (with-temp-buffer
        (insert-file-contents file)
        (let (models)
          (goto-char (point-min))
          ;; Match "- name: <model-id>" lines
          (while (re-search-forward "^- name:[ \t]+\\(.+\\)$" nil t)
            (let ((model-name (string-trim (match-string 1))))
              (when (and (not (string-empty-p model-name))
                         (not (string-prefix-p "#" model-name)))
                (push model-name models))))
          (nreverse models))))))

(defun aidermacs--model-id-match-p (configured-id model-id)
  "Check if CONFIGURED-ID matches MODEL-ID.
Uses exact match only - no fuzzy matching across providers."
  (string= configured-id model-id))

(defun aidermacs--find-litellm-prices-file ()
  "Find the local litellm prices file from Aider's installation."
  (or aidermacs--litellm-file-path-cache
      (when aidermacs-litellm-prices-file
        (let ((expanded (expand-file-name aidermacs-litellm-prices-file)))
          (when (file-exists-p expanded)
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

(defun aidermacs--fetch-openrouter-prices ()
  "Fetch model prices from OpenRouter API.
Returns an alist of model-id to ((input-price . val) (output-price . val)).
Model IDs are prefixed with \"openrouter/\"."
  (let ((attempt 0)
        result)
    (while (and (< attempt 2) (not result))
      (setq attempt (1+ attempt))
      (when (> attempt 1)
        (sleep-for 1.5))
      (condition-case _err
          (let* ((url-request-method "GET")
                 (url-request-extra-headers '(("Content-Type" . "application/json")))
                 (buf (url-retrieve-synchronously
                       "https://openrouter.ai/api/v1/models"
                       t nil 10))
                 response data)
            (unwind-protect
                (with-current-buffer buf
                  ;; Check HTTP status before attempting JSON parse
                  (let ((status (if (boundp 'url-http-response-status)
                                    url-http-response-status
                                  0)))
                    (if (or (zerop status) (>= status 400))
                        nil
                      (goto-char url-http-end-of-headers)
                      (let* ((json-object-type 'alist)
                             (json-key-type 'string)
                             (json-array-type 'list))
                        (setq response (json-read)))
                      (setq data (cdr (assoc "data" response)))
                      (dolist (model data)
                        (let* ((id (cdr (assoc "id" model)))
                               (pricing (cdr (assoc "pricing" model)))
                               (prompt (when pricing (cdr (assoc "prompt" pricing))))
                               (completion (when pricing (cdr (assoc "completion" pricing)))))
                          (when (and id prompt completion)
                            (push (cons (concat "openrouter/" (if (stringp id) id (format "%s" id)))
                                        `((input-price . ,(if (stringp prompt) (string-to-number prompt) prompt))
                                          (output-price . ,(if (stringp completion) (string-to-number completion) completion))))
                                  result))))
                      (setq result (nreverse result)))))
              (when (buffer-live-p buf)
                (kill-buffer buf))))
        (error
         (setq result nil))))
    result))

(defun aidermacs--build-price-index (litellm-prices)
  "Build fast lookup indexes from LITELLM-PRICES.
Returns a list (exact-hash family-hash provider-family-hash)."
  (let ((exact (make-hash-table :test 'equal :size (length litellm-prices)))
        (family (make-hash-table :test 'equal))
        (prov-fam (make-hash-table :test 'equal)))
    (dolist (entry litellm-prices)
      (let ((key (car entry))
            (info (cdr entry)))
        (when (stringp key)
          (puthash key info exact)
          (let* ((id (aidermacs--parse-model-identity key))
                 (fam (alist-get 'family id))
                 (prov (alist-get 'provider id)))
            (when fam
              (unless (gethash fam family)
                (puthash fam info family))
              (when prov
                (puthash (concat prov "/" fam) info prov-fam)))))))
    (list exact family prov-fam)))

(defun aidermacs--match-model-price-fast (model-id index)
  "Fast price lookup for MODEL-ID using prebuilt INDEX."
  (when index
    (let* ((identity (aidermacs--parse-model-identity model-id))
           (exact (nth 0 index))
           (family (nth 1 index))
           (prov-fam (nth 2 index)))
      (or (gethash model-id exact)
          (gethash (alist-get 'family identity) family)
          (let ((prov (alist-get 'provider identity))
                (fam (alist-get 'family identity)))
            (when (and prov fam)
              (gethash (concat prov "/" fam) prov-fam)))))))

(defun aidermacs--get-litellm-prices ()
  "Get model prices from litellm and OpenRouter, using cache if still valid."
  (if (and aidermacs--litellm-prices-cache
           aidermacs--litellm-prices-cache-timestamp
           (< (- (float-time) aidermacs--litellm-prices-cache-timestamp)
              aidermacs-litellm-prices-cache-duration))
      aidermacs--litellm-prices-cache
    (let ((litellm-prices (aidermacs--read-litellm-prices))
          (openrouter-prices (aidermacs--fetch-openrouter-prices)))
      ;; Merge: OpenRouter overrides litellm, new entries appended
      (let ((merged (copy-sequence litellm-prices)))
        (dolist (entry openrouter-prices)
          (let ((existing (assoc (car entry) merged)))
            (if existing
                (setcdr existing (cdr entry))
              (push entry merged))))
        (when merged
          (setq aidermacs--litellm-prices-cache merged)
          ;; If OpenRouter fetch failed, keep the cache short-lived so we retry
          ;; the network on the next model fetch instead of locking buggy state.
          (setq aidermacs--litellm-prices-cache-timestamp
                (if openrouter-prices
                    (float-time)
                  (- (float-time) (- aidermacs-litellm-prices-cache-duration 60)))))
        merged))))

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

(defun aidermacs--make-model-annotator (cheapest-models configured-models)
  "Create annotation function for the cheapest models.
CHEAPEST-MODELS is a list of (model . rank) from
`aidermacs--get-cheapest-models'.
CONFIGURED-MODELS is a list of model IDs that are user-configured."
  (let ((rank-map (make-hash-table :test 'equal))
        (configured-set (make-hash-table :test 'equal)))
    ;; Build rank map from cheapest models
    (dolist (entry cheapest-models)
      (puthash (alist-get 'id (car entry)) (cdr entry) rank-map))
    ;; Build configured set from all configured models
    (dolist (id configured-models)
      (puthash id t configured-set))
    (lambda (cand-id)
      (let ((rank (gethash cand-id rank-map))
            (is-configured (gethash cand-id configured-set)))
        (cond
         ((and rank is-configured)
          (format " [Rank %d - Cheapest] [Configured]" rank))
         (rank
          (format " [Rank %d - Cheapest]" rank))
         (is-configured
          " [Configured]")
         (t nil))))))

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
             ;; 1. Read configured models from settings file
             (configured-models (aidermacs--read-configured-models))
             ;; 2. Mark all models with configured-p property
             (marked-models
              (mapcar (lambda (m)
                        (let* ((id (alist-get 'id m))
                               (is-configured
                                (cl-some (lambda (cfg-id)
                                           (aidermacs--model-id-match-p cfg-id id))
                                         configured-models)))
                          (append m `((configured-p . ,is-configured)))))
                      aidermacs--cached-models))
             ;; 3. Filter/sort based on filter mode
             (filtered-models
              (pcase aidermacs-model-filter-mode
                ('configured-only
                 (if configured-models
                     (seq-filter (lambda (m) (alist-get 'configured-p m)) marked-models)
                   marked-models))
                ('configured-first
                 (if configured-models
                     (let ((configured (seq-filter (lambda (m) (alist-get 'configured-p m)) marked-models))
                           (others (seq-remove (lambda (m) (alist-get 'configured-p m)) marked-models)))
                       (append configured others))
                   marked-models))
                (_ marked-models)))
             ;; 4. Build annotator from cheapest and configured models
             (annotator (aidermacs--make-model-annotator
                         (aidermacs--get-cheapest-models filtered-models 500)
                         configured-models))
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
                      filtered-models)))
        (let ((model (completing-read
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
                  (aidermacs--send-command (format "/model %s" real-model)))))))))
    (quit (message "Model selection cancelled"))))

(defun aidermacs--parse-model-identity (model-id)
  "Parse MODEL-ID into canonical identity components.
Returns an alist with keys: provider, family, variant, full-id.
Examples:
  \"openai/gpt-4o-2024-08-06\" ->
    ((provider . \"openai\") (family . \"gpt-4o\") ...)
  \"claude-3-5-sonnet-20241022\" ->
    ((provider . nil) (family . \"claude-3-5-sonnet\") ...)"
  (unless (stringp model-id)
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
           (setq aidermacs--cached-models nil)
           (when callback (funcall callback)))
       (let* ((all-models
               (mapcar (lambda (line) (substring line 2))
                       (seq-filter (lambda (line) (string-prefix-p "- " line))
                                   (split-string aidermacs--current-output "\n" t))))
              (all-models-str (mapcar (lambda (m) (if (stringp m) m (format "%s" m))) all-models))
              (litellm-prices (aidermacs--get-litellm-prices))
              ;; Supplement with OpenRouter models from API
              (openrouter-ids (mapcar #'car
                                      (seq-filter (lambda (entry) (string-prefix-p "openrouter/" (car entry)))
                                                  litellm-prices)))
              (all-models-str (delete-dups (append all-models-str openrouter-ids)))
              (price-index (aidermacs--build-price-index litellm-prices))
              (models))
         (dolist (model-id all-models-str)
           (when (stringp model-id)
             (let* ((price-info (aidermacs--match-model-price-fast model-id price-index))
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
  (let ((prices-stale (or (null aidermacs--litellm-prices-cache-timestamp)
                          (>= (- (float-time) aidermacs--litellm-prices-cache-timestamp)
                              aidermacs-litellm-prices-cache-duration))))
    (if (and aidermacs--cached-models (not prices-stale))
        (aidermacs--select-model arg)
      (progn
        (when prices-stale
          (setq aidermacs--cached-models nil))
        (message "Fetching available models...")
        (aidermacs--get-available-models
         (lambda ()
           (message "Models fetched successfully")
           (aidermacs--select-model arg)))))))

(provide 'aidermacs-models)
;;; aidermacs-models.el ends here

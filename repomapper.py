#!/usr/bin/env python

"""
Standalone script to generate a repository map similar to Aider's repomap feature.

Based on code from the Aider project: https://github.com/paul-gauthier/aider

This script generates a repository map, which is a concise representation of
the codebase structure and relevant code snippets, intended to be included in
the context provided to a Large Language Model (LLM).

Default Behavior (LLM Context):
By default, the script performs a ranking algorithm (PageRank) based on code
dependencies and context provided via command-line arguments (--chat-files,
--mentioned-files, --mentioned-idents). It then selects the most relevant
code definitions and files, formats them into snippets using tree-sitter,
and prunes the result to fit within a specified token limit (--map-tokens).
This produces a concise, context-aware map suitable for LLM context windows.

--render-cache Behavior (Debugging/Inspection):
When the --render-cache flag is used, the script bypasses the ranking and
token-limiting steps. It loads all tags (definitions and references) from the
cache directory (.aider.tags.cache.v*) and renders them using the same
snippet-generation logic (TreeContext). This often results in showing large
portions, or even the entirety, of the cached files. This mode is useful for
inspecting the raw information captured by the tagger and the output of the
rendering engine, but its output is generally too large and unfiltered for
direct use as LLM context.

Install dependencies:
pip install networkx pygments grep-ast diskcache tiktoken tqdm gitignore_parser scipy
"""

import argparse
import math
import os
import shutil
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import tiktoken
from diskcache import Cache
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from tqdm import tqdm

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
try:
    # We still need get_language and get_parser from grep_ast.tsl
    from grep_ast.tsl import get_language, get_parser
except ImportError as e:
    print(
        "Error importing from grep_ast.tsl. Please ensure grep-ast and its dependencies are"
        " installed correctly."
    )
    print("Try: pip install grep-ast")
    sys.exit(f"ImportError: {e}")


# --- Constants and Definitions ---

Tag = namedtuple("Tag", "rel_fname fname line name kind".split())

SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError, OSError)

# Define a fixed cache directory name for this standalone script
TAGS_CACHE_DIR = ".aideremacs_repomap"

# --- File Reading Utility ---


def read_text(filename, encoding="utf-8", errors="ignore"):
    """Reads a file and returns its content."""
    try:
        with open(str(filename), "r", encoding=encoding, errors=errors) as f:
            return f.read()
    except FileNotFoundError:
        warnings.warn(f"{filename}: file not found error")
        return None
    except IsADirectoryError:
        warnings.warn(f"{filename}: is a directory")
        return None
    except OSError as err:
        warnings.warn(f"{filename}: unable to read: {err}")
        return None
    except UnicodeError as e:
        warnings.warn(f"{filename}: {e}")
        return None


# --- Relative Path Utility ---


def get_rel_fname(fname, root):
    """Gets the relative path of fname from the root."""
    try:
        return os.path.relpath(fname, root)
    except ValueError:
        # Handle cases where fname and root are on different drives (Windows)
        return fname


# --- Important Files Logic (from aider/special.py) ---

ROOT_IMPORTANT_FILES = [
    # Version Control
    ".gitignore",
    ".gitattributes",
    # Documentation
    "README",
    "README.md",
    "README.txt",
    "README.rst",
    "CONTRIBUTING",
    "CONTRIBUTING.md",
    "CONTRIBUTING.txt",
    "CONTRIBUTING.rst",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "CHANGELOG",
    "CHANGELOG.md",
    "CHANGELOG.txt",
    "CHANGELOG.rst",
    "SECURITY",
    "SECURITY.md",
    "SECURITY.txt",
    "CODEOWNERS",
    # Package Management and Dependencies
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "npm-shrinkwrap.json",
    "Gemfile",
    "Gemfile.lock",
    "composer.json",
    "composer.lock",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "build.sbt",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
    "mix.exs",
    "rebar.config",
    "project.clj",
    "Podfile",
    "Cartfile",
    "dub.json",
    "dub.sdl",
    # Configuration and Settings
    ".env",
    ".env.example",
    ".editorconfig",
    "tsconfig.json",
    "jsconfig.json",
    ".babelrc",
    "babel.config.js",
    ".eslintrc",
    ".eslintignore",
    ".prettierrc",
    ".stylelintrc",
    "tslint.json",
    ".pylintrc",
    ".flake8",
    ".rubocop.yml",
    ".scalafmt.conf",
    ".dockerignore",
    ".gitpod.yml",
    "sonar-project.properties",
    "renovate.json",
    "dependabot.yml",
    ".pre-commit-config.yaml",
    "mypy.ini",
    "tox.ini",
    ".yamllint",
    "pyrightconfig.json",
    # Build and Compilation
    "webpack.config.js",
    "rollup.config.js",
    "parcel.config.js",
    "gulpfile.js",
    "Gruntfile.js",
    "build.xml",
    "build.boot",
    "project.json",
    "build.cake",
    "MANIFEST.in",
    # Testing
    "pytest.ini",
    "phpunit.xml",
    "karma.conf.js",
    "jest.config.js",
    "cypress.json",
    ".nycrc",
    ".nycrc.json",
    # CI/CD
    ".travis.yml",
    ".gitlab-ci.yml",
    "Jenkinsfile",
    "azure-pipelines.yml",
    "bitbucket-pipelines.yml",
    "appveyor.yml",
    "circle.yml",
    ".circleci/config.yml",
    ".github/dependabot.yml",
    "codecov.yml",
    ".coveragerc",
    # Docker and Containers
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.override.yml",
    # Cloud and Serverless
    "serverless.yml",
    "firebase.json",
    "now.json",
    "netlify.toml",
    "vercel.json",
    "app.yaml",
    "terraform.tf",
    "main.tf",
    "cloudformation.yaml",
    "cloudformation.json",
    "ansible.cfg",
    "kubernetes.yaml",
    "k8s.yaml",
    # Database
    "schema.sql",
    "liquibase.properties",
    "flyway.conf",
    # Framework-specific
    "next.config.js",
    "nuxt.config.js",
    "vue.config.js",
    "angular.json",
    "gatsby-config.js",
    "gridsome.config.js",
    # API Documentation
    "swagger.yaml",
    "swagger.json",
    "openapi.yaml",
    "openapi.json",
    # Development environment
    ".nvmrc",
    ".ruby-version",
    ".python-version",
    "Vagrantfile",
    # Quality and metrics
    ".codeclimate.yml",
    "codecov.yml",
    # Documentation
    "mkdocs.yml",
    "_config.yml",
    "book.toml",
    "readthedocs.yml",
    ".readthedocs.yaml",
    # Package registries
    ".npmrc",
    ".yarnrc",
    # Linting and formatting
    ".isort.cfg",
    ".markdownlint.json",
    ".markdownlint.yaml",
    # Security
    ".bandit",
    ".secrets.baseline",
    # Misc
    ".pypirc",
    ".gitkeep",
    ".npmignore",
]

# Normalize the lists once
NORMALIZED_ROOT_IMPORTANT_FILES = set(os.path.normpath(path) for path in ROOT_IMPORTANT_FILES)


def is_important(file_path):
    file_name = os.path.basename(file_path)
    dir_name = os.path.normpath(os.path.dirname(file_path))
    normalized_path = os.path.normpath(file_path)

    # Check for GitHub Actions workflow files
    if dir_name == os.path.normpath(".github/workflows") and file_name.endswith((".yml", ".yaml")):
        return True

    return normalized_path in NORMALIZED_ROOT_IMPORTANT_FILES


def filter_important_files(file_paths):
    """
    Filter a list of file paths to return only those that are commonly important in codebases.

    :param file_paths: List of file paths to check (relative to repo root)
    :return: List of file paths that match important file patterns
    """
    # For standalone script, assume paths are relative to the root already
    return list(filter(is_important, file_paths))


# --- RepoMap Class (adapted from aider/repomap.py) ---


class RepoMap:
    warned_files = set()

    def __init__(
        self,
        root,
        map_tokens=4096,
        verbose=False,
        tokenizer_name="cl100k_base",  # Default tokenizer for gpt-4, gpt-3.5
        force_refresh=False,
    ):
        self.verbose = verbose
        self.root = os.path.abspath(root)
        self.max_map_tokens = map_tokens
        self.force_refresh = force_refresh

        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception as e:
            print(f"Error initializing tokenizer '{tokenizer_name}': {e}")
            print("Please ensure tiktoken is installed: pip install tiktoken")
            sys.exit(1)

        self.load_tags_cache()

        self.tree_cache = {}
        self.tree_context_cache = {}
        self.map_processing_time = 0

        if self.verbose:
            print(f"RepoMap initialized for root: {self.root}")
            print(f"Using map token limit: {self.max_map_tokens}")

    def token_count(self, text):
        """Counts tokens using the tiktoken tokenizer."""
        # Simplified token counting for standalone script
        if not isinstance(text, str):
            text = str(text) # Ensure text is string
        # Aider uses a more complex sampling method for large text,
        # but direct encoding is fine for typical map sizes here.
        return len(self.tokenizer.encode(text))

    def get_repo_map(self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None):
        """Generates the repository map string."""
        if self.max_map_tokens <= 0:
            print("Map tokens set to 0, skipping map generation.")
            return ""
        if not other_files and not chat_files: # Need at least some files to map
            print("No files provided for repository map.")
            return ""
        # Combine chat_files and other_files for processing, but keep track of chat_files for ranking
        all_files = set(chat_files) | set(other_files)

        start_time = time.time()
        try:
            files_listing = self.get_ranked_tags_map_uncached(
                chat_files, other_files, self.max_map_tokens, mentioned_fnames, mentioned_idents
            )
        except RecursionError:
            print("ERROR: Recursion error during map generation. Repo might be too large.")
            return ""
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during map generation: {e}")
            import traceback
            traceback.print_exc()
            return ""
        end_time = time.time()
        self.map_processing_time = end_time - start_time

        if not files_listing:
            print("No map content generated.")
            return ""

        if self.verbose:
            num_tokens = self.token_count(files_listing)
            print(f"Repo Map generated: {num_tokens} tokens, took {self.map_processing_time:.2f}s")

        # Determine prefix based on whether chat_files were provided
        if chat_files:
            other = "other "
        else:
            other = ""

        repo_content = f"Repository Map ({other}files):\n" # Use a consistent prefix

        if self.verbose:
            num_tokens = self.token_count(files_listing)
        repo_content += files_listing
        return repo_content

    def tags_cache_error(self, original_error=None):
        """Handle SQLite errors by trying to recreate cache, falling back to dict if needed"""
        if self.verbose and original_error:
            warnings.warn(f"Tags cache error: {str(original_error)}")

        if isinstance(getattr(self, "TAGS_CACHE", None), dict):
            return # Already using dict cache

        path = Path(self.root) / TAGS_CACHE_DIR

        # Try to recreate the cache
        try:
            print(f"Attempting to recreate tags cache at {path}...")
            # Delete existing cache dir
            if path.exists():
                shutil.rmtree(path)

            # Try to create new cache
            new_cache = Cache(path)

            # Test that it works
            test_key = "test"
            new_cache[test_key] = "test"
            _ = new_cache[test_key]
            del new_cache[test_key]

            # If we got here, the new cache works
            self.TAGS_CACHE = new_cache
            print("Successfully recreated tags cache.")
            return

        except SQLITE_ERRORS as e:
            # If anything goes wrong, warn and fall back to dict
            warnings.warn(
                f"Unable to use disk cache at {path}, falling back to in-memory cache. Error: {e}"
            )
            if self.verbose:
                warnings.warn(f"Cache recreation error details: {str(e)}")

        self.TAGS_CACHE = dict() # Fallback to in-memory dict

    def load_tags_cache(self):
        """Loads the tags cache from disk or initializes it."""
        path = Path(self.root) / TAGS_CACHE_DIR
        try:
            self.TAGS_CACHE = Cache(path)
            # Basic check to see if cache is usable
            _ = len(self.TAGS_CACHE)
            if self.verbose:
                print(f"Using disk cache at {path}")
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
        except Exception as e:
            warnings.warn(f"Unexpected error loading cache {path}: {e}. Using in-memory cache.")
            self.TAGS_CACHE = dict()

    def save_tags_cache(self):
        """Saves the tags cache (no-op for diskcache, it saves automatically)."""
        pass # diskcache handles saving

    def get_mtime(self, fname):
        """Gets the modification time of a file."""
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            warnings.warn(f"File not found error getting mtime: {fname}")
            return None

    def get_tags(self, fname, rel_fname):
        """Gets tags for a file, using the cache if possible."""
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        cache_key = fname
        try:
            # Use get with default=None to avoid KeyError if key doesn't exist
            val = self.TAGS_CACHE.get(cache_key, default=None)
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            val = self.TAGS_CACHE.get(cache_key, default=None) # Retry after potential cache reset
        except Exception as e:
             warnings.warn(f"Unexpected error reading from cache for {fname}: {e}")
             val = None # Treat as cache miss

        # Check if cache hit is valid and not forced to refresh
        if (not self.force_refresh and
            val is not None and
            isinstance(val, dict) and
            val.get("mtime") == file_mtime):
            try:
                # Ensure data exists and is iterable
                cached_data = val.get("data", [])
                return list(cached_data) if cached_data is not None else []
            except SQLITE_ERRORS as e:
                self.tags_cache_error(e)
                # Retry getting data after potential cache reset
                val = self.TAGS_CACHE.get(cache_key, default={})
                cached_data = val.get("data", [])
                return list(cached_data) if cached_data is not None else []
            except Exception as e:
                warnings.warn(f"Unexpected error accessing cached data for {fname}: {e}")
                # Fall through to re-generate tags

        # Cache miss or invalid data
        if self.verbose:
            print(f"Cache miss for {rel_fname}, generating tags...")
        data = list(self.get_tags_raw(fname, rel_fname))

        # Update the cache with both mtime and current time
        try:
            cache_entry = {
                "mtime": file_mtime,
                "map_time": time.time(),
                "data": data
            }
            self.TAGS_CACHE[cache_key] = cache_entry
            self.save_tags_cache()
            if self.verbose:
                print(f"Updated cache for {rel_fname} with mtime {file_mtime}")
        except SQLITE_ERRORS as e:
            self.tags_cache_error(e)
            # Try saving again if cache was reset to dict
            if isinstance(self.TAGS_CACHE, dict):
                 self.TAGS_CACHE[cache_key] = {"mtime": file_mtime, "map_time": time.time(), "data": data}
        except Exception as e:
            warnings.warn(f"Unexpected error writing to cache for {fname}: {e}")

        return data

    def get_tags_raw(self, fname, rel_fname):
        """Generates tags for a file using tree-sitter and pygments."""
        lang = filename_to_lang(fname)
        if not lang:
            return

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            # Don't stop execution, just skip the file
            warnings.warn(f"Skipping file {fname}: Can't get tree-sitter parser for language '{lang}'. Error: {err}")
            return

        # Find the path to the SCM query file
        query_scm_path = get_scm_fname(lang)
        query_scm = None

        if query_scm_path:
            try:
                query_scm = query_scm_path.read_text(encoding='utf-8')
            except Exception as e:
                warnings.warn(f"Error reading SCM file {query_scm_path}: {e}")
                query_scm = None # Ensure fallback if read fails

        if not query_scm:
             warnings.warn(f"No SCM query file found or loaded for language '{lang}' for file {fname}. Relying on pygments.")


        code = read_text(fname) # Use the utility function
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        saw_defs = False
        saw_refs = False

        # Run the tags queries if available
        if query_scm:
            try:
                query = language.query(query_scm)
                captures = query.captures(tree.root_node)

                # Assumes modern grep-ast returning a dict {tag_name: [nodes]}
                all_nodes = []
                for tag_name, nodes in captures.items():
                    all_nodes += [(node, tag_name) for node in nodes]

                for node, tag_name in all_nodes:
                    if tag_name.startswith("name.definition."):
                        kind = "def"
                        saw_defs = True
                    elif tag_name.startswith("name.reference."):
                        kind = "ref"
                        saw_refs = True
                    else:
                        continue

                    try:
                        name_text = node.text.decode("utf-8")
                    except (AttributeError, UnicodeDecodeError):
                        continue # Skip nodes without valid text

                    yield Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        name=name_text,
                        kind=kind,
                        line=node.start_point[0],
                    )
            except Exception as e:
                warnings.warn(f"Error running tree-sitter query for {fname}: {e}")


        # If we saw only defs (or no SCM query ran), use pygments for refs
        if saw_defs and not saw_refs or not query_scm:
            if self.verbose and not query_scm:
                 print(f"Using pygments for refs in {rel_fname} (no SCM query)")
            elif self.verbose and saw_defs and not saw_refs:
                 print(f"Using pygments to supplement refs in {rel_fname}")

            try:
                lexer = guess_lexer_for_filename(fname, code)
                tokens = list(lexer.get_tokens(code))
                # Filter for names (identifiers)
                name_tokens = [token[1] for token in tokens if token[0] in Token.Name]

                for token_text in name_tokens:
                    yield Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        name=token_text,
                        kind="ref",
                        line=-1, # Line number unknown from pygments tokens
                    )
            except Exception as e:
                warnings.warn(f"Error using pygments for {fname}: {e}")
                return # Stop processing this file if pygments fails

    def get_ranked_tags(self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents):
        """Ranks tags based on PageRank of the dependency graph, personalized by context."""
        import networkx as nx

        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)
        personalization = dict() # For PageRank personalization

        all_fnames = set(chat_fnames) | set(other_fnames)
        chat_rel_fnames = set(get_rel_fname(fname, self.root) for fname in chat_fnames)
        mentioned_rel_fnames = set(get_rel_fname(fname, self.root) for fname in mentioned_fnames)

        print("Scanning files and building graph...")
        # Use tqdm for progress if available
        fnames_iter = tqdm(sorted(list(all_fnames)), desc="Scanning", unit="file") if 'tqdm' in sys.modules else sorted(list(all_fnames))

        # Calculate base personalization value
        num_nodes_estimate = len(all_fnames)
        personalize_base = 100 / num_nodes_estimate if num_nodes_estimate > 0 else 1

        for fname in fnames_iter:
            if self.verbose:
                print(f"Processing {fname}")

            try:
                file_ok = Path(fname).is_file()
            except OSError:
                file_ok = False

            if not file_ok:
                if fname not in self.warned_files:
                    warnings.warn(f"Repo-map can't include {fname} (not a file or inaccessible)")
                    self.warned_files.add(fname)
                continue

            rel_fname = get_rel_fname(fname, self.root)

            # Set personalization score for context files
            if rel_fname in chat_rel_fnames or rel_fname in mentioned_rel_fnames:
                 personalization[rel_fname] = personalize_base

            tags = list(self.get_tags(fname, rel_fname)) # Use cached tags

            if not tags: # Skip files with no tags
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)
                elif tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        # If no references found (e.g., only C++ defs), use defines as refs for graph
        if not references and defines:
            print("No references found, using definitions for graph linking.")
            references = {k: list(v) for k, v in defines.items()}

        idents = set(defines.keys()).intersection(set(references.keys()))
        if not idents:
            print("No common identifiers found between definitions and references. Map may be incomplete.")
            # Still proceed to rank files based on structure if possible

        G = nx.MultiDiGraph()

        print("Building dependency graph...")
        idents_iter = tqdm(idents, desc="Linking", unit="ident") if 'tqdm' in sys.modules else idents
        for ident in idents_iter:
            definers = defines[ident]

            # Adjust weight multiplier based on whether the identifier was mentioned
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"): # Penalize private/internal identifiers slightly
                mul = 0.1
            else:
                mul = 1

            # Basic weighting: sqrt of reference count
            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # Aider includes self-loops, keep for consistency
                    # if referencer == definer: continue

                    # Scale down so high freq (low value) mentions don't dominate
                    weight = math.sqrt(num_refs)
                    G.add_edge(referencer, definer, weight=mul * weight, ident=ident) # Apply multiplier here

        if not G.edges():
             print("Graph has no edges. Ranking will be based on file structure only.")
             # Add all files as nodes so PageRank doesn't fail
             for fname in all_fnames:
                 rel_fname = get_rel_fname(fname, self.root)
                 if not G.has_node(rel_fname):
                     G.add_node(rel_fname)


        print("Running PageRank...")
        pers_args = dict()
        if personalization:
             # Use personalization if context was provided
             pers_args = dict(personalization=personalization, dangling=personalization)
             if self.verbose:
                 print(f"Using personalization: {personalization}")

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            warnings.warn("ZeroDivisionError during PageRank. Graph might be disconnected.")
            # Fallback: Rank nodes equally if PageRank fails, respecting personalization if possible
            num_nodes = G.number_of_nodes()
            if num_nodes > 0:
                base_rank = 1.0 / num_nodes
                ranked = {node: personalization.get(node, base_rank) for node in G.nodes()}
                # Normalize if personalization was used
                if personalization:
                    total_rank = sum(ranked.values())
                    if total_rank > 0:
                         ranked = {node: r / total_rank for node, r in ranked.items()}
                    else: # Handle case where total rank is zero
                         ranked = {node: base_rank for node in G.nodes()}
            else:
                ranked = {}
        except Exception as e:
            warnings.warn(f"Error during PageRank: {e}. Map quality may be affected.")
            ranked = {} # Empty ranking on other errors


        # Distribute rank from files to the definitions within them
        ranked_definitions = defaultdict(float)
        if G.edges(): # Only distribute if graph has structure
            print("Distributing rank to definitions...")
            nodes_iter = tqdm(G.nodes(), desc="Distributing", unit="node") if 'tqdm' in sys.modules else G.nodes()
            for src in nodes_iter:
                src_rank = ranked.get(src, 0) # Use .get for safety
                # Calculate total weight of outgoing edges *from this source*
                total_weight = sum(data.get("weight", 0) for _src, _dst, data in G.out_edges(src, data=True))

                if total_weight > 0:
                    for _src, dst, data in G.out_edges(src, data=True):
                        ident = data.get("ident")
                        weight = data.get("weight", 0)
                        if ident: # Ensure ident exists
                            # Use the rank calculated by PageRank for the source node
                            rank_share = src_rank * weight / total_weight
                            ranked_definitions[(dst, ident)] += rank_share
        else:
             print("Skipping rank distribution (no graph edges).")


        # Collect ranked tags
        ranked_tags_list = []
        # Sort definitions by rank
        sorted_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: (x[1], x[0])
        )

        # Add definitions based on their rank, excluding those in chat_fnames
        fnames_already_included_from_defs = set()
        for (fname, ident), _rank in sorted_definitions:
            if fname in chat_rel_fnames: # Exclude definitions from files already in chat
                 continue
            # Add all Tag objects associated with this definition key
            def_tags = definitions.get((fname, ident), set())
            ranked_tags_list.extend(list(def_tags))
            fnames_already_included_from_defs.add(fname)

        # Add remaining files (not in chat) based on their overall PageRank score
        # These files might be important structurally even if their specific defs weren't top-ranked
        rel_other_fnames = set(get_rel_fname(fname, self.root) for fname in other_fnames)
        sorted_files_by_rank = sorted(ranked.items(), reverse=True, key=lambda item: item[1])

        for fname, _rank in sorted_files_by_rank:
            # Only consider files that are in 'other_fnames' and not already included via definitions
            if fname in rel_other_fnames and fname not in fnames_already_included_from_defs:
                # Represent these files as tuples to distinguish from Tag objects
                ranked_tags_list.append((fname,))
                # Remove from set to avoid adding again below
                rel_other_fnames.remove(fname)


        # Add any remaining 'other_fnames' that weren't ranked at all (e.g., disconnected components)
        for fname in sorted(list(rel_other_fnames)): # Sort for consistent output
             if fname not in fnames_already_included_from_defs:
                 ranked_tags_list.append((fname,))


        return ranked_tags_list

    def get_ranked_tags_map_uncached(
        self, chat_fnames, other_fnames, max_map_tokens, mentioned_fnames=None, mentioned_idents=None
    ):
        """Generates the map string from ranked tags, fitting it into the token limit."""
        if not mentioned_fnames: mentioned_fnames = set()
        if not mentioned_idents: mentioned_idents = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        # Prioritize important files from 'other_fnames'
        other_rel_fnames = sorted(set(get_rel_fname(fname, self.root) for fname in other_fnames))
        special_fnames = filter_important_files(other_rel_fnames)

        # Get filenames already represented by ranked tags (these are already filtered to exclude chat_fnames)
        ranked_tags_fnames = set(tag.rel_fname for tag in ranked_tags if isinstance(tag, Tag))
        ranked_files_only = set(tag[0] for tag in ranked_tags if isinstance(tag, tuple))
        all_ranked_fnames = ranked_tags_fnames.union(ranked_files_only)


        # Prepare special files to be potentially added
        # Add them as file-only tuples `(fname,)`
        special_fnames_to_add = [(fn,) for fn in special_fnames if fn not in all_ranked_fnames]

        # Combine: special files first, then the ranked tags/files
        combined_ranked_items = special_fnames_to_add + ranked_tags

        print(f"Total ranked items (tags/files) considered for map: {len(combined_ranked_items)}")
        print("Finding optimal map size for token limit...")

        num_items = len(combined_ranked_items)
        lower_bound = 0
        upper_bound = num_items
        best_tree = ""
        best_tree_tokens = 0

        # Clear tree cache for this run
        self.tree_cache = dict()

        # Estimate initial middle point based on average tokens per item (heuristic)
        # Assume ~25 tokens per tag/file entry as a rough starting point
        initial_middle_estimate = min(int(max_map_tokens / 25), num_items) if num_items > 0 else 0
        middle = initial_middle_estimate

        # Binary search to find the best number of items to include
        iterations = 0
        max_iterations = int(math.log2(num_items)) + 5 if num_items > 0 else 0 # Safety limit

        while lower_bound <= upper_bound and iterations < max_iterations:
            iterations += 1
            current_items = combined_ranked_items[:middle]
            if not current_items:
                # If middle is 0, check if we need to increase lower bound
                if num_items > 0:
                    lower_bound = middle + 1
                    middle = int((lower_bound + upper_bound) / 2)
                    continue
                else:
                    break # No items to process


            print(f"  Trying {middle}/{num_items} items...")
            # Pass chat_rel_fnames to to_tree to ensure they are excluded from the output map
            chat_rel_fnames = set(get_rel_fname(fname, self.root) for fname in chat_fnames)
            tree = self.to_tree(current_items, chat_rel_fnames)
            num_tokens = self.token_count(tree)
            print(f"    Tokens: {num_tokens}/{max_map_tokens}")

            # Check if this is the best result so far that fits
            if num_tokens <= max_map_tokens:
                if num_tokens > best_tree_tokens:
                    best_tree = tree
                    best_tree_tokens = num_tokens
                    print(f"    New best map found ({best_tree_tokens} tokens)")

                # If it fits, try including more items
                lower_bound = middle + 1
            else:
                # If it doesn't fit, try including fewer items
                upper_bound = middle - 1

            # Adjust middle for next iteration
            middle = int((lower_bound + upper_bound) / 2)

            # Optimization: If the best map is already close to the limit, stop early
            if best_tree_tokens > max_map_tokens * 0.95:
                 print("    Best map is close to token limit, stopping search.")
                 break


        print(f"Selected map size: {best_tree_tokens} tokens")
        return best_tree

    def render_tree(self, abs_fname, rel_fname, lois):
        """Renders code snippets for a file using TreeContext."""
        mtime = self.get_mtime(abs_fname)
        if mtime is None: return f"# Error: Could not get mtime for {rel_fname}\n"

        # Cache key includes filename, lines of interest, and modification time
        lois_tuple = tuple(sorted(list(set(lois)))) # Ensure unique, sorted lines
        key = (rel_fname, lois_tuple, mtime)

        if key in self.tree_cache:
            return self.tree_cache[key]

        # Check context cache
        cached_context_info = self.tree_context_cache.get(rel_fname)
        if cached_context_info and cached_context_info.get("mtime") == mtime:
            context = cached_context_info["context"]
        else:
            # Need to create or update context
            code = read_text(abs_fname)
            if code is None:
                return f"# Error: Could not read {rel_fname}\n"
            if not code.endswith("\n"):
                code += "\n"

            try:
                context = TreeContext(
                    rel_fname,
                    code,
                    color=False, # No color for plain text map
                    line_number=False,
                    child_context=False,
                    last_line=False,
                    margin=0,
                    mark_lois=False,
                    loi_pad=0,
                    show_top_of_file_parent_scope=False,
                )
                self.tree_context_cache[rel_fname] = {"context": context, "mtime": mtime}
            except Exception as e:
                 warnings.warn(f"Error creating TreeContext for {rel_fname}: {e}")
                 return f"# Error processing {rel_fname}\n"


        # Configure and run TreeContext for the current lines of interest
        try:
            context.lines_of_interest = set(lois) # Use the current set of lines
            context.add_context() # Determine context lines based on LOIs
            res = context.format() # Format the output
        except Exception as e:
            warnings.warn(f"Error formatting TreeContext for {rel_fname} lines {lois}: {e}")
            res = f"# Error formatting {rel_fname}\n"


        # Store the rendered output in the tree cache
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags_or_files, chat_rel_fnames):
        """Formats the selected ranked tags/files into the final map string, excluding chat_rel_fnames."""
        if not tags_or_files:
            return ""

        output = ""
        # Group tags by file
        grouped_tags = defaultdict(list)
        files_only = []

        for item in tags_or_files:
            # Explicitly skip any item whose filename is in chat_rel_fnames
            if isinstance(item, Tag):
                if item.rel_fname in chat_rel_fnames:
                    continue
                grouped_tags[item.rel_fname].append(item)
            elif isinstance(item, tuple) and len(item) == 1:
                if item[0] in chat_rel_fnames:
                    continue
                # This is a file-only entry
                files_only.append(item[0])
            else:
                 warnings.warn(f"Unexpected item type in ranked list: {type(item)}")


        # Process files with tags first (already filtered for chat_rel_fnames)
        sorted_fnames_with_tags = sorted(grouped_tags.keys())

        for rel_fname in sorted_fnames_with_tags:
            file_tags = grouped_tags[rel_fname]
            abs_fname = file_tags[0].fname # Get abs path from the first tag
            lois = [tag.line for tag in file_tags if tag.line >= 0] # Collect line numbers

            if not lois: # If only file-level refs were found (line -1)
                 output += "\n" + rel_fname + "\n" # Just list the filename
            else:
                output += "\n"
                output += rel_fname + ":\n"
                rendered_tree = self.render_tree(abs_fname, rel_fname, lois)
                output += rendered_tree

        # Add files that were ranked but had no specific tags selected (already filtered for chat_rel_fnames)
        sorted_files_only = sorted(files_only)
        for rel_fname in sorted_files_only:
             # Check if already added via grouped_tags (already filtered, so this check is less critical but safe)
             if rel_fname not in grouped_tags:
                 output += "\n" + rel_fname + "\n"


        # Truncate long lines (safety measure)
        output = "\n".join([line[:200] for line in output.splitlines()]) # Increased limit slightly
        if output: # Add trailing newline if not empty
             output += "\n"

        return output


# --- Helper Functions (from aider/repomap.py) ---


# Common binary/non-source file extensions to exclude
BINARY_EXTS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', '.svg',
    # Media
    '.mp3', '.mp4', '.mov', '.avi', '.mkv', '.wav',
    # Archives
    '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    # Documents
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    # Other binaries
    '.exe', '.dll', '.so', '.o', '.a', '.class', '.jar'
}




def get_scm_fname(lang):
    """
    Finds the tree-sitter query file for a given language,
    assuming it's in ./queries/tree-sitter-languages/ relative to this script.
    """
    try:
        # Get the directory containing this script (map.py)
        script_dir = Path(__file__).parent.resolve()
        # Construct the path to the query file
        query_path = script_dir / "queries" / "tree-sitter-languages" / f"{lang}-tags.scm"

        if query_path.is_file():
            return query_path
        else:
            # Optional: Add verbose logging here if needed
            # print(f"DEBUG: SCM file not found at expected path: {query_path}")
            return None
    except Exception as e:
        warnings.warn(f"Error trying to locate SCM file for {lang}: {e}")
        return None


class RepoMapper:
    def __init__(self, root_dir, map_tokens=4096, tokenizer="cl100k_base", verbose=False, force_refresh=False):
        self.root = os.path.abspath(root_dir)
        self.map_tokens = map_tokens
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.force_refresh = force_refresh
        self.repo_mapper = RepoMap(
            root=self.root,
            map_tokens=self.map_tokens,
            verbose=self.verbose,
            tokenizer_name=self.tokenizer,
            force_refresh=self.force_refresh,
        )
        # Initialize map generation timestamp
        self.map_generation_time = time.time()

    def _is_gitignored(self, path):
        """Check if path matches any .gitignore rules."""
        try:
            from gitignore_parser import parse_gitignore
            gitignore_path = os.path.join(self.root, '.gitignore')
            if os.path.exists(gitignore_path):
                gitignore = parse_gitignore(gitignore_path)
                return gitignore(path)
        except ImportError:
            if self.verbose:
                print("Note: gitignore_parser not installed, .gitignore checking disabled")
        return False

    def _find_src_files(self, directory):
        """Finds all files in a directory recursively, excluding binaries."""
        if not os.path.isdir(directory):
            if os.path.exists(directory):
                if os.path.splitext(directory)[1].lower() in BINARY_EXTS:
                    return []
                return [directory]
            warnings.warn(f"Input path is not a directory or file: {directory}")
            return []

        src_files = []
        if self.verbose:
            print(f"Scanning directory: {directory}")
        for root, dirs, files in os.walk(directory, topdown=True):
            # Filter directories
            dirs[:] = [
                d for d in dirs
                if not (
                    d.startswith('.') or  # hidden dirs
                    d in {'node_modules', 'vendor', 'build', 'dist', '__pycache__', '.venv', 'env'}
                )
            ]

            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if (
                    ext in BINARY_EXTS or  # binary files
                    file.startswith('.') or     # hidden files
                    self._is_gitignored(file_path)  # gitignored files
                ):
                    continue

                src_files.append(file_path)

        if self.verbose:
            print(f"Found {len(src_files)} potential source files.")
        return src_files

    def generate_map(self, chat_files=None, mentioned_files=None, mentioned_idents=None, force_refresh=None):
        """Generate repository map with optional context files/identifiers

        Args:
            chat_files: List of files in chat context
            mentioned_files: List of mentioned files
            mentioned_idents: Set of mentioned identifiers
            force_refresh: If True, ignores cache and regenerates all files
        """
        if chat_files is None:
            chat_files = []
        if mentioned_files is None:
            mentioned_files = []
        if mentioned_idents is None:
            mentioned_idents = set()
        if force_refresh is not None:
            self.force_refresh = force_refresh

        # Update map generation time
        self.map_generation_time = time.time()
        if self.verbose:
            print(f"Map generation started at: {self.map_generation_time}")

        # Resolve paths relative to root
        def resolve_path(p):
            abs_p = os.path.abspath(os.path.join(self.root, p))
            if not os.path.exists(abs_p):
                if self.verbose:
                    warnings.warn(f"Context file not found: {p} (resolved to {abs_p})")
                return None
            return abs_p

        chat_files_abs = [p for p in (resolve_path(f) for f in chat_files) if p]
        mentioned_files_abs = [p for p in (resolve_path(f) for f in mentioned_files) if p]
        mentioned_idents = set(mentioned_idents)

        # Find all files in repo
        all_repo_files = self._find_src_files(self.root)
        if not all_repo_files:
            if self.verbose:
                print(f"No source files found in directory: {self.root}")
            return ""

        # Determine other_files by removing chat_files
        chat_files_set = set(chat_files_abs)
        other_files_abs = [f for f in all_repo_files if f not in chat_files_set]

        # Generate and return map content
        map_content = self.repo_mapper.get_repo_map(
            chat_files=chat_files_abs,
            other_files=other_files_abs,
            mentioned_fnames=mentioned_files_abs,
            mentioned_idents=mentioned_idents,
        )

        if self.verbose:
            print(f"Map generation completed at: {time.time()}")
        return map_content

    def render_cache(self):
        """Render all cached tags without ranking/selection"""
        cache_path = Path(self.root) / TAGS_CACHE_DIR
        if not cache_path.exists() or not cache_path.is_dir():
            if self.verbose:
                print(f"Error: Cache directory not found at {cache_path}")
            return ""

        try:
            cache = Cache(cache_path)
            if self.verbose:
                print(f"Found {len(cache)} items in cache.")

            all_tags = []
            all_cached_fnames = set()

            # Collect all tags and filenames from cache
            for key in cache.iterkeys():
                try:
                    abs_fname = key
                    if not os.path.exists(abs_fname) or os.path.isdir(abs_fname):
                        continue

                    all_cached_fnames.add(abs_fname)
                    cached_item = cache.get(key)
                    if cached_item and isinstance(cached_item, dict) and "data" in cached_item:
                        all_tags.extend(cached_item.get("data", []))
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Error processing cache key {key}: {e}")

            # Create temporary RepoMap for rendering
            temp_mapper = RepoMap(
                root=self.root,
                map_tokens=1_000_000,  # Large token limit for full cache dump
                verbose=self.verbose,
                tokenizer_name=self.tokenizer,
            )

            # Prepare items for rendering
            tag_fnames = set(tag.rel_fname for tag in all_tags)
            all_cached_rel_fnames = set(get_rel_fname(fname, self.root) for fname in all_cached_fnames)
            files_only = all_cached_rel_fnames - tag_fnames

            items_to_render = list(all_tags) + [(fname,) for fname in sorted(files_only)]
            rendered_map = temp_mapper.to_tree(items_to_render, chat_rel_fnames=set())

            cache.close()
            return rendered_map

        except Exception as e:
            if self.verbose:
                print(f"Error rendering cache: {e}")
            return ""


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate a repository map similar to Aider's repomap feature."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regenerate all files, ignoring cache timestamps",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="The root directory of the repository/project to map.",
    )
    parser.add_argument(
        "--map-tokens",
        type=int,
        default=4096,
        help="The target maximum number of tokens for the generated map.",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="The name of the tiktoken tokenizer to use.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during map generation.",
    )
    parser.add_argument(
        "--output",
        help="Optional file path to write the map to.",
    )
    parser.add_argument(
        "--render-cache",
        action="store_true",
        help="Render all tags found in the cache.",
    )
    parser.add_argument(
        "--chat-files",
        nargs='*',
        default=[],
        help="Files considered 'in the chat' for context.",
    )
    parser.add_argument(
        "--mentioned-files",
        nargs='*',
        default=[],
        help="Files explicitly mentioned for context.",
    )
    parser.add_argument(
        "--mentioned-idents",
        nargs='*',
        default=[],
        help="Identifiers explicitly mentioned for context.",
    )

    args = parser.parse_args()

    mapper = RepoMapper(
        root_dir=args.dir,
        map_tokens=args.map_tokens,
        tokenizer=args.tokenizer,
        verbose=args.verbose,
        force_refresh=args.force_refresh
    )

    if args.render_cache:
        content = mapper.render_cache()
        if content:
            print("\n--- Rendered Cache Map ---")
            print(content)
            print("--- End Rendered Cache Map ---")
        else:
            print("Failed to render cache.")
        return

    content = mapper.generate_map(
        chat_files=args.chat_files,
        mentioned_files=args.mentioned_files,
        mentioned_idents=args.mentioned_idents
    )

    if content:
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Repository map written to: {args.output}")
            except IOError as e:
                print(f"Error writing map: {e}")
                print("\n--- Repository Map ---")
                print(content)
                print("--- End Repository Map ---")
        else:
            print("\n--- Repository Map ---")
            print(content)
            print("--- End Repository Map ---")
    else:
        print("Failed to generate repository map.")



if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start local dev server (live reload at http://localhost:1313)
hugo server

# Build for production
hugo --gc --minify

# Create a new post (generates TOML front matter from archetype)
hugo new posts/YYYYMMDD-slug.md
```

No test suite or linter is configured. Hugo version used in CI is **0.150.0 extended**.

The PaperMod theme is a git submodule at `themes/PaperMod`. After cloning, run:
```bash
git submodule update --init --recursive
```

## Architecture

This is a **Hugo static site** (personal blog/portfolio) deployed to GitHub Pages via `.github/workflows/hugo.yaml` on every push to `main`.

### Content Sections

| Section | Path | Purpose |
|---|---|---|
| Posts | `content/posts/` | Technical blog articles (AI/ML focus) |
| Projects | `content/projects/` | Portfolio of engineering projects |
| Talks | `content/talks/` | Conference talks and presentations |
| About | `content/about-me/` | Bio page |

Posts may live directly under `content/posts/` or in subdirectories (e.g., `content/posts/cv-techniques/`, `content/posts/nbooks/`). Jupyter notebooks (`.ipynb`) can exist alongside Markdown posts in subdirectories.

### Front Matter Conventions

**Posts** use TOML (`+++`):
```toml
+++
title = 'Post Title'
date = 2025-01-01T00:00:00+01:00
draft = false
tags = ['tag1', 'tag2']
metaDescription = 'SEO description'
recommendations = ['20241029-attention', 'cv-techniques/20240615-cv-techniques']
[cover]
    image = "/posts/YYYY/image.png"
+++
```

**Projects** and **Talks** use YAML (`---`):
```yaml
# Projects
---
title: "Project Name"
date: 2026-01-01
summary: "One-line summary shown in hero"
tags: ["rag", "nlp"]
tech: ["Python", "FastAPI", "Docker"]
cover:
  image: "/projects/slug/image.png"
demo: "https://..."
repo: "https://..."
---

# Talks
---
title: "Talk Title"
event: "Conference Name"
location: "City, Country"
date: 2024-10-01
talkType: "Talk"
tags: ["llm", "rag"]
video: "https://www.youtube.com/watch?v=VIDEO_ID"
---
```

Post filenames follow the pattern `YYYYMMDD-slug.md`. Static images for posts go in `static/posts/YYYY/`.

### `recommendations` Field

Posts support a `recommendations` front matter list that renders a sidebar/inline section linking to related posts. Values are matched by filename stem (e.g., `20241029-attention`) or relative path for posts in subdirectories (e.g., `cv-techniques/20240615-cv-techniques`). Resolution logic is in `layouts/partials/recommendations.html`.

### Custom Layouts

PaperMod's templates are overridden/extended in `layouts/`:

- `layouts/_default/single.html` — adds ToC sidebar, recommendations sidebar, PDF/ePub download buttons, and a scroll-aware JS that animates sidebar position
- `layouts/projects/single.html` — hero with tech/tag pills, cover image, repo/demo action buttons
- `layouts/talks/single.html` — hero with event metadata, embedded YouTube player (auto-extracts video ID), watch button

### Shortcodes

| Shortcode | Usage |
|---|---|
| `{{< callout-info title="Note" >}}` | Blue info box |
| `{{< callout-curiosity title="..." >}}` | Curiosity/tip box |
| `{{< callout-seealso title="..." >}}` | See-also reference box |
| `{{< image src="..." alt="..." >}}` | Centered image |
| `{{< figure src="..." caption="..." width="50%" align="center" >}}` | Hugo built-in with caption |

### Math

LaTeX math is enabled globally (`params.math: true` in `hugo.yaml`). Use goldmark passthrough delimiters:
- Inline: `\( ... \)` or `\[ ... \]`
- Block: `$$ ... $$`

### CSS

Custom styles live in `assets/css/common/` and are loaded via `hugo.yaml` params:
- `css/projects.css` — project hero, tech/tag pills, action buttons
- `css/talks.css` — talk hero, video embed, type badge
- `callouts.css`, `post-single.css`, etc. in `assets/css/common/`

`assets/css/core/` contains theme-level overrides (do not edit PaperMod files directly in `themes/`; override in `assets/` or `layouts/`).

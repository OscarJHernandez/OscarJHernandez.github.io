# Oscar J. Hernandez — Personal Site

Personal portfolio and blog. Built with Jekyll, styled with SCSS (compiled via Gulp), and deployed to GitHub Pages.

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| **Node.js** | 20+ | [nodejs.org](https://nodejs.org/) |
| **Ruby** | 3.2+ | [ruby-lang.org](https://www.ruby-lang.org/) or `rbenv` / `rvm` |
| **Bundler** | latest | `gem install bundler` |

---

## Local Setup (first time)

```bash
# 1. Clone the repo
git clone https://github.com/OscarJHernandez/OscarJHernandez.github.io.git
cd OscarJHernandez.github.io

# 2. Install Node dependencies (Gulp, Sass, etc.)
npm install

# 3. Install Ruby dependencies (Jekyll, html-proofer, etc.)
bundle install
```

---

## Running Locally

```bash
# Compile SCSS + JS, build Jekyll, and start BrowserSync with live reload
npm run serve
```

The site is served at **http://localhost:3000** and rebuilds automatically on any file change.

If you only want to compile assets without starting a server:

```bash
npm run build
```

If you only want to run Jekyll directly (no BrowserSync):

```bash
bundle exec jekyll serve
# → http://localhost:4000
```

---

## Project Structure

```
_config.yml          # Site settings (name, email, analytics ID)
_includes/           # Reusable HTML partials (header, footer, etc.)
_layouts/            # Page templates (default, blog, post, resume)
_posts/              # Blog posts (YYYY-MM-DD-title.md)
src/
  styles/            # SCSS source files (compiled → assets/css/main.css)
  js/                # JS source (compiled → assets/js/main.js)
assets/              # Compiled/static output — do not edit directly
```

---

## Writing a Blog Post

### 1. Create a file in `_posts/`

Filename format: `_posts/YYYY-MM-DD-short-title.md`

Example: `_posts/2026-05-10-gradient-descent.md`

### 2. Add front matter

```yaml
---
layout: post
title: "Your Post Title"
date: 2026-05-10
category: Mathematics
description: One sentence shown on the blog index page.
---
```

| Field | Required | Notes |
|---|---|---|
| `layout` | yes | Always `post` |
| `title` | yes | Shown in sidebar and post header |
| `date` | yes | Must match the filename date |
| `category` | yes | Groups posts in sidebar & index (e.g. `Mathematics`, `System Design`, `Physics`) |
| `description` | no | Short summary shown on blog index |

### 3. Write content in Markdown

**Math** (MathJax is enabled):

```markdown
Inline: $E = mc^2$

Block:
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$
```

**Code blocks:**

````markdown
```python
def hello():
    return "world"
```
````

---

## Customization

### Site info

Edit `_config.yml`:

```yaml
username: Oscar J. Hernandez
user_title: "Data Scientist & Engineer"
email: you@example.com
google-analytics:
  id: "G-XXXXXXXXXX"   # leave empty to disable
```

### Colors

Edit `src/styles/_vars.scss`, then recompile:

```bash
npm run build
```

Current palette:

| Variable | Value | Used for |
|---|---|---|
| `$main` | `#1a222c` | Hero background, headings |
| `$link` | `#2563eb` | Links, active states |
| `$link-hover` | `#1d4ed8` | Link hover |
| `$sec` | `#4B5664` | Secondary text |

---

## Automated Testing (CI)

Three test jobs run automatically on every push and pull request via GitHub Actions:

| Job | What it checks |
|---|---|
| **Build + HTMLProofer** | Jekyll build succeeds; all internal links, images, and scripts resolve |
| **SCSS Lint** | `src/styles/**/*.scss` passes Stylelint (`stylelint-config-standard-scss`) |
| **Accessibility** | Home, blog index, and resume pass WCAG 2AA via pa11y-ci |

The deploy to GitHub Pages only runs after all three jobs pass on `master`.

### Run tests locally

```bash
# SCSS lint
npm run lint:scss

# HTMLProofer (requires Jekyll build first)
bundle exec jekyll build
bundle exec htmlproofer ./_site --disable-external --checks "Links,Images,Scripts"

# Accessibility
npx serve ./_site -l 4000
npx pa11y-ci --config .pa11yci
```

The accessibility check runs Headless Chrome via Pa11y. CI-specific browser launch settings live in `/.pa11yci`.

---

## Deployment

Pushing to `master` triggers the CI workflow. If all tests pass, the site is automatically deployed to GitHub Pages. No manual steps needed.

To deploy manually (skipping CI):

1. Go to **Actions → Deploy Jekyll site to Pages → Run workflow**

---

## License

MIT

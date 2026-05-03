# Particle Jekyll Theme

This is a simple and minimalist template for Jekyll designed for developers that want to show of their portfolio.

The Theme features:

- Gulp
- SASS
- Sweet Scroll
- Particle.js
- BrowserSync
- Font Awesome and Devicon icons
- Google Analytics
- Info Customization

## Basic Setup

1. [Install Jekyll](http://jekyllrb.com)
2. Fork the [Particle Theme](https://github.com/nrandecker/particle/fork)
3. Clone the repo you just forked.
4. Edit `_config.yml` to personalize your site.

## Site and User Settings

You have to fill some informations on `_config.yml` to customize your site.

```
# Site settings
description: A blog about lorem ipsum dolor sit amet
baseurl: "" # the subpath of your site, e.g. /blog/
url: "http://localhost:3000" # the base hostname & protocol for your site

# User settings
username: Lorem Ipsum
user_description: Anon Developer at Lorem Ipsum Dolor
user_title: Anon Developer
email: anon@anon.com
twitter_username: lorem_ipsum
github_username:  lorem_ipsum
gplus_username:  lorem_ipsum
```

**Don't forget to change your url before you deploy your site!**

## Color and Particle Customization
- Color Customization
  - Edit the sass variables
- Particle Customization
  - Edit the json data in particle function in app.js
  - Refer to [Particle.js](https://github.com/VincentGarreau/particles.js/) for help

---

## Writing a Blog Post

All posts live in the `_posts/` folder.

### 1. Create a file

The filename **must** follow this exact format:

```
_posts/YYYY-MM-DD-short-title.md
```

Example: `_posts/2026-05-10-gradient-descent.md`

### 2. Add front matter

Every post starts with a YAML block at the top:

```yaml
---
layout: post
title: "Your Post Title"
date: 2026-05-10
category: Mathematics
description: One sentence shown on the blog index page.
---
```

**Fields:**
| Field | Required | Notes |
|---|---|---|
| `layout` | yes | Always `post` |
| `title` | yes | Shown in sidebar and post header |
| `date` | yes | Must match the filename date |
| `category` | yes | Groups posts in sidebar & index (e.g. `Mathematics`, `System Design`, `Physics`) |
| `description` | no | Short summary shown on blog index |

### 3. Write content

After the front matter, write standard Markdown.

**Math** (MathJax is enabled):
```
Inline: $E = mc^2$

Block:
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$
```

**Code blocks:**
````
```python
def hello():
    return "world"
```
````

**Table of contents** — add `{:toc}` after a heading:
```markdown
## Contents
{:toc}
```

### 4. Preview

```bash
npm run serve
```

The site rebuilds automatically on save. Blog is at **http://localhost:3000/blog/**

## Running the blog in local

In order to compile the assets and run Jekyll on local you need to follow those steps:

- Install [NodeJS](https://nodejs.org/)
- Run `npm install`
- Run `npm run build` to compile assets (and Jekyll when Ruby is set up)
- Run `npm run serve` (or `npx gulp`) for BrowserSync after `bundle install`

## Running Locally with Jekyll:
* Install gem bundler:
$ gem install bundler
$ bundle install
$ sudo apt-get install ruby2.3 ruby2.3-dev 

* Run locally
$ bundle exec jekyll serve

## Questions

Having any issues file a [GitHub Issue](https://github.com/nrandecker/particle/issues/new).

## License

This theme is free and open source software, distributed under the The MIT License. So feel free to use this Jekyll theme anyway you want.

## Credits

This theme was partially designed with the inspiration from these fine folks
- [Willian Justen](https://github.com/willianjusten/will-jekyll-template)
- [Vincent Garreau](https://github.com/VincentGarreau/particles.js/)

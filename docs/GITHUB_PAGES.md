# GitHub Pages Setup

This repository is prepared so that GitHub Pages can serve the project homepage directly from the `docs/` folder.

## Recommended Settings

In the GitHub repository settings:

1. Open `Settings`
2. Open `Pages`
3. Set `Source` to `Deploy from a branch`
4. Set `Branch` to `main`
5. Set folder to `/docs`
6. Save

GitHub Pages will then publish:

- `docs/index.html` as the homepage
- `docs/styles.css` as the site stylesheet
- `docs/paper-assets/` as the local paper figure asset folder

## Current Site Content

- homepage / overview
- links to GitHub and Hugging Face resources
- method summary
- performance snapshot
- citation block

## Notes

- `docs/index.md` is kept as repository documentation content and does not need to be removed.
- `index.html` takes precedence as the Pages landing page.
- You can continue editing the site later by modifying only the files in `docs/`.

## Suggested Future Updates

- add the arXiv / paper link after release
- add author names and affiliations
- add a Hugging Face collection link
- add a project banner or teaser figure

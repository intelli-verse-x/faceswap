# IntelliVerse-X fork notes

This repository is a **public fork** of [deepfakes/faceswap](https://github.com/deepfakes/faceswap) owned by the [`intelli-verse-x`](https://github.com/intelli-verse-x) organization.

## Purpose of this fork

- **Research** and experimentation with face-swap techniques
- **Offline HQ training** (extract → train → convert) for **owned / consented** brand characters
- Optional **convert-only** use of pre-trained brand models in a **separate** GPU worker

## What this fork is not

- **Not** imported by Content Factory (`content-factory` is MIT; this tree is **GPL-3.0**)
- **Not** the default runtime path for CF video jobs (those stay minutes-scale; classic Faceswap training is hours–days)
- **Not** a tool for non-consent, celebrity abuse, illicit, or hidden synthetic identity fraud

## License pin

- Upstream license remains **GPL-3.0** — see [`LICENSE`](LICENSE)
- Do **not** remove, weaken, or strip the GPL notice
- Do **not** vendor this tree into proprietary CF packages; keep Faceswap as a **separate** program/service

## Ethical use / consent only

Aligned with the upstream [Manifesto](README.md#manifesto) and [faceswap.dev manifesto](https://faceswap.dev/manifesto/):

1. **Consent required** — only faces you own or have clear written permission to use
2. **No illicit or inappropriate use**
3. **No hiding** that a synthetic face-lock / swap was applied when content is published for product use
4. **Zero tolerance** for non-consent deepfakes

IntelliVerse product path (Content Factory) must enforce consent attestation and disclosure in the calling system; this repo alone does not replace that gate.

## Relationship to Content Factory

| Layer | Repo / service | Role |
|-------|----------------|------|
| CF runtime | `content-factory` (MIT) | Pipelines, plan → APPROVE, one-shot face-lock worker client |
| This fork | `intelli-verse-x/faceswap` (GPL-3.0) | Offline HQ train / research |
| Worker | Separate container/process | May call convert-only or one-shot tools over HTTP/files — not an in-process CF import |

## CODEOWNERS

See [`.github/CODEOWNERS`](.github/CODEOWNERS).

arXiv submission package — MicroVLA
"Learning Location, Not Looking: A Placement-Memorization Audit and
 Ten-Episode Repair at the 30M Scale"

CONTENTS (self-contained; builds with pdflatex alone, no bibtex pass needed):
  main.tex        the paper
  main.bbl        pre-compiled bibliography (arXiv prefers .bbl over .bib)
  visuals/*.png   the three figures, referenced as visuals/...

VERIFIED: builds standalone inside this directory, 9 pages, 0 errors,
0 undefined citations. Figure paths are package-relative, not repo-relative.

TO SUBMIT: tar czf arxiv.tgz main.tex main.bbl visuals && upload to arxiv.org.
Suggested categories: cs.RO (primary), cs.LG (cross-list).

NOT SUBMITTED. Posting a preprint is the author's decision and needs the
author's credentials; this package only removes the preparation work from it.

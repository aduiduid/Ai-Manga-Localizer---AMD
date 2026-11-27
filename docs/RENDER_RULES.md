# Renderer change rules

Learned the hard way on 2026-11-22, from three separate mistakes in one session.
Every rule below exists because something went wrong in a specific, recorded way.

## 1. Never decide colour from brightness alone

An absolute threshold on paper brightness cannot separate an anti-aliasing rim
from coloured lettering, because both are dark and saturated. A rule that tried
("bright paper plus dark saturated ink is a fringe") shipped in 0.13.15 and
silently repainted **230 units** on the sample book: every violet line of
dialogue, the red lettering, the cover title.

Decisions about whether a colour belongs to the glyphs are **proportional**: ask
how much of the crop the colour covers. `INK_SHARE_MIN` (4%) is that rule. A 4 px
rim around black bars measures 3.8% and never becomes a candidate; real lettering
is tens of percent. When adding a colour rule, state it as a proportion or do not
add it.

## 2. Colour tests compare pixels, not objects

The brightness rule shipped with a green test suite. Its own test asserted on the
`TextStyle` object, and the assertion still passed after the decision moved to a
different branch: the object said `ink=(84,40,117)` while the renderer painted
black.

Colour regressions are tested by rendering real crops and comparing **drawn
pixels** with **source pixels** -- `tests/test_render_colour_golden.py`, ten
fixtures of real art, asserting the drawn glyphs keep both the source's channel
spread and its hue. Object-level assertions are still useful for logic, but they
do not count as colour coverage.

**Prove a new colour test can fail.** Temporarily reintroduce the bug it guards,
watch it go red, then restore. A test that has never been seen failing is a
comment.

## 3. Never argue about colour from a picture

Two people in one session misread a screenshot: one saw red lettering where the
source was dark green, another read a downscaled render as black where the pixels
were violet. Downscaling, resampling and memory all lie about hue.

Use `scripts/pixel_probe.py` before changing colour code and after:

```
python scripts/pixel_probe.py <source-or-book> --unit p015u01 --page 00000015 \
    --rendered <rendered book>
```

It prints the paper level, the glyph colour, the channel spread and the hue name
for the same region in both images. A claim about colour has to survive that
output.

## 4. Stale output looks like a bug

The cover credit "had its bottom edge cut off" for two rounds. It had not: the
shipped page predated the fix that removed its plate, and the current code
reproduces the source art with **0 changed pixels out of 905600**.

Before debugging a rendering defect reported against a folder, check when that
folder was written against the last renderer change:

- `git -C app log -1 --format=%ad --date=iso`
- file mtime of the reported page
- re-render the page from its cache and diff it against the shipped file

`tests/rerender_book.py`-style re-renders (see `_home/tmp` history) take ~2 s per
page from a warm cache and settle the question immediately.

## 5. A decorative overlay is not evidence of what is under it

The plate under spilled type was reported for three rounds as "red lettering
turned black". The lettering was never red: source glyphs measure (72,89,71),
channel spread 17. What had changed was the two pink wedges either side of that
line, which the invisible white card painted over.

The lesson: when a pixel looks wrong, find the *step* that moved it before
theorising about the colour path. _home/tmp/isolate_corners.py answered it in
one run by disabling one unit and re-rendering -- the erase step had not touched
the wedges at all, the plate had.

Corollary: a rule that decides whether to paint a decorative element must be
argued from what that element is *for* (legibility: luminance separation), not
from how busy the area looks. Every "is this card dirty" measure failed on 746
cards, because the artwork beside the line is inside the card and always reads as
dirt.

## 6. Record what you chose not to fix

Some defects have no zero-cost fix. The current known limit: when a box holds a
coloured stamp or heart across black type, `glyph_ink` can report the decoration
instead of the lettering -- 6 units of 1350 on the sample book. It shows up as a
wrong-*hue* translation, never as a missing erase.

Half-fixing these is worse than documenting them: the attempt to pick "the
darkest vivid cluster" changed no output at all (the lettering cluster sits
below `INK_VIVID`), so it was reverted rather than kept as dead complexity.
Known limits live in `CHANGELOG.md` and in the UI hint, where the user meets
them.

The second one, recorded 2026-11-27: white lettering with a saturated rim on a
coloured balloon survives the erase. Reading the white core instead of the rim
does not fix it, because the core is neutral and a neutral reading is dropped to
`None` before the mask is built -- the erase then falls back to a brightness
seed. Every selection rule that was tried also changed hundreds of units that
were already correct, because a balloon's own highlight and white lettering are
the same colour; only the detector box tells them apart, and on page 409 that box
spans several balloons. Four rules were measured and all rejected (74.2%, 67.8%,
75.0% and 88.0% of their new mask landing on paper, against an acceptance of
30%). A defect that needs a different box, not a different threshold, is a known
limit.

# Characterization — measuring the detector before trusting it

Before closing the loop, OWL-ViT was characterized the way a sensor would be:
where it succeeds, where it fails, and how far its best true detection sits above
its best false one (the **margin**). The trust gate in the closed loop is set from
these measurements, not from a guessed threshold.

## Prompt sensitivity

Same frame, same cube, different text prompts — the score and margin swing widely:

| Prompt | Cube score | Best false | Margin | Verdict |
|--------|-----------|-----------|--------|---------|
| `a partially red object` | 0.312 | 0.202 | **+0.110** | ROBUST |
| `a partially red object; color is very important` | 0.126 | 0.065 | +0.061 | THIN |
| `red cube` | 0.041 | 0.007 | +0.035 | (below floor) |

Counterintuitively, the *more elaborated* prompt scored the cube **lower** and
thinner. OWL-ViT matches a whole-phrase embedding against the image (the CLIP
lineage), so added abstract words ("color is very important") drift the match away
from the target rather than sharpening it. And shape words hurt: `red cube` scored
worst of all — a plain synthetic primitive doesn't read as a 3-D "cube" to a model
trained on real photos. The plain, generic `a partially red object` won. Negation
("not the arm") had no effect at all — CLIP does similarity, not logic.

`a partially red object` was adopted as the operating prompt.

## Non-monotonic confidence vs. target size

Enlarging the cube (by moving the camera closer) does **not** monotonically
improve detection:

| Cube size | Cube score | Margin |
|-----------|-----------|--------|
| ~18 px (far) | 0.016 | — |
| ~40 px (mid) | 0.312 | **+0.110** (cube wins) |
| ~90 px (close) | 0.223 | **−0.050** (arm wins) |

Confidence peaks at an intermediate size and *degrades* when the camera gets close
enough to also enlarge the distractor (the arm). This is the effect that drives
the two demo outcomes: the loop acts at the mid framing and refuses at the close
one.

## Latency

OWL-ViT inference runs ~230 ms per frame on the GPU used here — the dominant term
in the perception→command path, well above the ~15 Hz frame arrival and the
sim's render/readback delays. For the deliberative one-shot-plus-nudge loop this
is fine; it's recorded because it's the number that would matter if the loop were
ever run at speed.

Raw console runs backing the prompt table: [`prompt_sweep.txt`](prompt_sweep.txt).

# Demos — the gate deciding

Two runs of the same controller on the same scene, at two camera framings. The
only difference is how confidently OWL-ViT can tell the cube from the arm — and
the gate decides accordingly.

| Run | Framing | Cube score | Arm score | Margin | Gate | Result |
|-----|---------|-----------|-----------|--------|------|--------|
| [`gate_open_success/`](gate_open_success/) | mid (~40 px cube) | 0.312 | 0.202 | **+0.110** | OPEN | arm drives, 44 cm closed |
| [`gate_refuse/`](gate_refuse/) | close (~90 px cube) | 0.223 | 0.273 | **−0.050** | REFUSE | no motion |

Each folder holds the input frame, the detector's annotated output (boxes +
scores), and the console trace.

**Success:** at the mid framing the cube outscores the arm by a clear margin, the
gate opens, and the controller nudges the arm — closing the gripper-to-cube
distance from 0.99 m to 0.55 m over ten steps.

**Refusal:** move the camera closer and the cube grows to ~90 px — but so does the
arm, and OWL-ViT then ranks the *arm* higher than the cube. The margin goes
negative, and the gate refuses to move the arm on a detection it can't trust. The
cube was correctly located (on-target score 0.22, not a miss); the gate declines
because the detection is genuinely ambiguous, not because it failed to find the
target.

That the bigger cube detects *worse* is the non-monotonic effect measured in
[`../characterization/`](../characterization/) — which is why the trust threshold
is set from data.

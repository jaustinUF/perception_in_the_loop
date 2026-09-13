"""
Stage 5E.3 - OWL-ViT off-loop detector : the 5F dress rehearsal proper.

WHAT THIS IS
    A PURE-PYTHON, OFF-LOOP detector. No ROS, no rclpy, no sim -- it runs against
    a SAVED PNG frame (the one stage5e2_image_subscriber.py writes to disk from
    /camera/image_raw). It feeds that frame to OWL-ViT with a TEXT prompt, prints
    EVERY surviving detection with its score, overlays the boxes, and saves an
    annotated PNG.

    This is 5E step 2: run the detector OFF-LOOP and CHARACTERIZE it, exactly the
    read-only grounding check 5F will do with ER 2 -- but with a local, free, fast
    model. Same text-in / box-out interaction shape as ER 2, so this is a true
    dress rehearsal. NO motion, NO control loop.

WHY OFF-LOOP / WHY A SAVED FILE
    One variable at a time. Proving "can OWL-ViT box this cube" is a SEPARATE
    question from "does the live pipeline work." Running against a static PNG
    isolates the detector completely: any success/failure is the detector's and
    the image's, not the transport's. Wiring the detector to the live topic is a
    LATER change (and step 3, closing the loop, is optional).

ACQUISITION / ANALYSIS SPLIT (the standing discipline)
    This script ACQUIRES: it runs the model and writes artifacts --
      - an annotated PNG (boxes + scores overlaid on the frame)
      - a plain-text detections dump (every box, every score, machine-readable)
    Judgment happens on those artifacts (eyeball the PNG; a notebook can load the
    dump to tabulate/plot). The script does NOT pre-judge: it prints ALL
    detections above a LOW threshold so we see the score DISTRIBUTION, rather than
    tuning a threshold up front. WHERE to set the threshold is a characterization
    FINDING, not a default baked in here.

MODEL
    google/owlvit-base-patch32 -- weights cached at first use (already pulled).
    Runs on GPU if torch.cuda.is_available() (the RTX 5000 Ada), else CPU. The
    inference-latency number this prints is a REAL characterization datum: it is
    what OWL-ViT adds, in series, on top of the 5B 3-frame pipeline latency and
    the 5E.2 ~15 Hz readback-bound publish rate. (Off-loop here, so it does not
    yet STACK -- but the number is the one that will stack when/if a loop closes.)

USAGE (native-Jazzy terminal, system Python 3.12, where transformers/torch live)
    python3 stage5e3_owlvit_detect.py <frame.png> [--prompt "red cube"] \
            [--threshold 0.1] [--outdir <dir>]

    Examples:
      python3 stage5e3_owlvit_detect.py stage5e2_first_frame.png
      python3 stage5e3_owlvit_detect.py frame.png --prompt "a small red block"
      python3 stage5e3_owlvit_detect.py frame.png --threshold 0.05

    The prompt and threshold are CLI args (not edited in code) precisely BECAUSE
    they are the characterization knobs -- vary them freely without touching the
    script. "cube" vs "square", "red cube" vs "block", threshold sweeps: all just
    re-runs with different args, each writing its own annotated PNG.

    NOTE ON MULTIPLE PROMPTS: pass several comma-separated phrases to detect them
    all at once, each labeled -- e.g. --prompt "red cube, robot arm, gripper".
    OWL-ViT scores each phrase independently against the image.
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

MODEL_ID = "google/owlvit-base-patch32"


def parse_args():
    p = argparse.ArgumentParser(
        description="OWL-ViT off-loop detector for a saved camera frame."
    )
    p.add_argument("frame", help="path to the saved PNG frame to run against")
    p.add_argument(
        "--prompt",
        default="red cube",
        help='text prompt(s), comma-separated for multiple '
             '(default: "red cube")',
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="score threshold for KEEPING a detection (default: 0.1, "
             "deliberately low so we SEE the score distribution)",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="where to write annotated PNG + detections dump "
             "(default: alongside the input frame)",
    )
    p.add_argument(
        "--target-xy",
        default=None,
        help="approx pixel center of the TRUE target 'x,y' (e.g. 640,567). "
             "If given, every box is classed on-target vs off-target and the "
             "MARGIN (best on-target score - best off-target score) is reported. "
             "This is the real characterization number: a hit with a big margin "
             "is robust; a hit with a small margin is fragile (arm nearly tied).",
    )
    p.add_argument(
        "--target-tol",
        type=float,
        default=60.0,
        help="pixels: a box counts as ON-target if its center is within this "
             "distance of --target-xy (default: 60)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve paths and validate the input frame exists (fast-fail guard,
    # same discipline as the sim node's Franka-USD check).
    # ------------------------------------------------------------------
    frame_path = os.path.abspath(args.frame)
    if not os.path.isfile(frame_path):
        sys.exit(f"[error] frame not found: {frame_path}")

    outdir = args.outdir or os.path.dirname(frame_path)
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(frame_path))[0]

    # Prompts: split on comma, strip whitespace, drop empties.
    prompts = [s.strip() for s in args.prompt.split(",") if s.strip()]
    if not prompts:
        sys.exit("[error] no non-empty prompt given")

    # ------------------------------------------------------------------
    # Device. Report it -- the inference time below is only meaningful with
    # the device it ran on (GPU number != CPU number).
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_name = (
        torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    )
    print(f"[info] device: {device} ({dev_name})")
    print(f"[info] frame:  {frame_path}")
    print(f"[info] prompt(s): {prompts}")
    print(f"[info] keep-threshold: {args.threshold}")

    # ------------------------------------------------------------------
    # Load model + processor (weights are cached; this is load, not download).
    # Timed separately from inference so the two costs don't blur.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    processor = OwlViTProcessor.from_pretrained(MODEL_ID)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    t_load = time.perf_counter() - t0
    print(f"[info] model loaded in {t_load:.2f} s")

    # ------------------------------------------------------------------
    # Load the frame. Convert to RGB explicitly -- the subscriber saved rgb8,
    # but a defensive convert costs nothing and guarantees 3 channels.
    # ------------------------------------------------------------------
    image = Image.open(frame_path).convert("RGB")
    W, H = image.size
    print(f"[info] frame size: {W}x{H}")

    # ------------------------------------------------------------------
    # INFERENCE. OWL-ViT wants a list of prompt-lists (one prompt-list per
    # image); we pass one image, so [prompts].
    # ------------------------------------------------------------------
    inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()          # so the timer captures real GPU work
    t_infer = time.perf_counter() - t0
    print(f"[info] inference in {t_infer * 1000:.1f} ms "
          f"(this is the number that STACKS on 5B/5E latency if a loop closes)")

    # ------------------------------------------------------------------
    # Post-process to image-pixel boxes. target_sizes is (height, width).
    # We pass threshold=0.0 to the processor so we get EVERYTHING, then filter
    # ourselves -- keeps the score distribution visible (see the low-score note).
    # ------------------------------------------------------------------
    target_sizes = torch.tensor([[H, W]], device=device)
    # transformers renamed this method to post_process_grounded_object_detection
    # (signature verified against the installed package). threshold=0.0 keeps ALL
    # candidates so we filter ourselves and see the full score distribution.
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.0,
        text_labels=[prompts],
    )[0]

    boxes = results["boxes"].cpu().numpy()     # (N, 4) xyxy in pixels
    scores = results["scores"].cpu().numpy()   # (N,)
    labels = results["labels"].cpu().numpy()   # (N,) index into prompts

    # Sort by score descending so the printout reads best-first.
    order = np.argsort(-scores)
    boxes, scores, labels = boxes[order], scores[order], labels[order]

    kept = scores >= args.threshold
    n_total = len(scores)
    n_kept = int(kept.sum())

    # ------------------------------------------------------------------
    # MARGIN REPORT (the real characterization number). If --target-xy was
    # given, classify every box as ON-target (its center is within --target-tol
    # of the true target center) or OFF-target, then report the best score of
    # each and the MARGIN between them. This does NOT depend on the threshold, so
    # it works whether or not anything cleared it -- fixing the "when a box fires,
    # you lose the runner-up" gap. A big margin = robust detection; a small margin
    # = fragile (a false positive, e.g. the arm, is nearly tied with the cube).
    # ------------------------------------------------------------------
    tgt = None
    if args.target_xy:
        try:
            tx, ty = (float(v) for v in args.target_xy.split(","))
            tgt = (tx, ty)
        except Exception:
            sys.exit(f"[error] --target-xy must be 'x,y' (got '{args.target_xy}')")

    def _box_center(b):
        return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))

    best_on = None   # (score, idx)
    best_off = None
    if tgt is not None:
        for i in range(n_total):
            cx, cy = _box_center(boxes[i])
            dist = ((cx - tgt[0]) ** 2 + (cy - tgt[1]) ** 2) ** 0.5
            on = dist <= args.target_tol
            # scores are sorted desc, so the FIRST on/off we meet is that side's best
            if on and best_on is None:
                best_on = (scores[i], i)
            elif (not on) and best_off is None:
                best_off = (scores[i], i)
            if best_on is not None and best_off is not None:
                break

    # ------------------------------------------------------------------
    # PRINT every kept detection. This is the acquisition read: raw numbers,
    # best-first, so the score distribution is visible at a glance.
    # ------------------------------------------------------------------
    print(f"\n[result] {n_total} raw candidates; "
          f"{n_kept} above threshold {args.threshold}:")
    if n_kept == 0:
        print("  (none) -- the detector found nothing it is confident about.")
        # Still show the top few raw candidates so we see HOW close it got.
        show = min(5, n_total)
        if show:
            print(f"  top {show} raw candidates (below threshold), for context:")
            for i in range(show):
                x0, y0, x1, y1 = boxes[i]
                print(f"    score {scores[i]:.4f}  '{prompts[labels[i]]}'  "
                      f"box [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")
    else:
        for i in range(n_kept):
            x0, y0, x1, y1 = boxes[i]
            bw, bh = x1 - x0, y1 - y0
            print(f"  score {scores[i]:.4f}  '{prompts[labels[i]]}'  "
                  f"box [{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]  "
                  f"({bw:.0f}x{bh:.0f} px)")

    # --- The margin read (always printed when --target-xy is given) ----------
    if tgt is not None:
        print(f"\n[margin] target ~({tgt[0]:.0f},{tgt[1]:.0f}) "
              f"tol {args.target_tol:.0f} px:")
        if best_on is not None:
            bi = best_on[1]
            bx = boxes[bi]
            print(f"  best ON-target : score {best_on[0]:.4f}  "
                  f"'{prompts[labels[bi]]}'  "
                  f"box [{bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f}]")
        else:
            print("  best ON-target : (no box centered on the target at all)")
        if best_off is not None:
            oi = best_off[1]
            ox = boxes[oi]
            print(f"  best OFF-target: score {best_off[0]:.4f}  "
                  f"'{prompts[labels[oi]]}'  "
                  f"box [{ox[0]:.0f},{ox[1]:.0f},{ox[2]:.0f},{ox[3]:.0f}]  "
                  f"<- best FALSE positive")
        else:
            print("  best OFF-target: (none)")
        if best_on is not None and best_off is not None:
            margin = best_on[0] - best_off[0]
            verdict = ("ROBUST" if margin > 0.10 else
                       "THIN" if margin > 0.03 else
                       "FRAGILE (false positive nearly tied)")
            print(f"  MARGIN         : {margin:+.4f}  [{verdict}]")
        elif best_on is not None:
            print(f"  MARGIN         : n/a (no off-target box) "
                  f"-- on-target score stands alone")

    # ------------------------------------------------------------------
    # OVERLAY kept boxes and SAVE the annotated PNG (artifact #1).
    # ------------------------------------------------------------------
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # A small palette so different prompts get different colors.
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 128, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    for i in range(n_kept):
        x0, y0, x1, y1 = boxes[i]
        color = palette[int(labels[i]) % len(palette)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag = f"{prompts[labels[i]]} {scores[i]:.2f}"
        # Label background for legibility over a busy image.
        ty = max(0, y0 - 12)
        draw.text((x0 + 2, ty), tag, fill=color, font=font)

    thr_tag = f"{args.threshold:.2f}".replace(".", "p")
    prompt_tag = "_".join(prompts[0].split())[:20]   # first prompt, filename-safe
    annotated_path = os.path.join(
        outdir, f"{stem}__det_{prompt_tag}_thr{thr_tag}.png"
    )
    annotated.save(annotated_path)
    print(f"\n[info] annotated frame -> {annotated_path}")

    # ------------------------------------------------------------------
    # WRITE the detections dump (artifact #2) -- machine-readable, so a notebook
    # can load many runs and tabulate. One row per KEPT detection.
    # ------------------------------------------------------------------
    dump_path = os.path.join(
        outdir, f"{stem}__det_{prompt_tag}_thr{thr_tag}.txt"
    )
    with open(dump_path, "w") as f:
        f.write(f"# frame: {frame_path}\n")
        f.write(f"# size_wh: {W} {H}\n")
        f.write(f"# prompts: {prompts}\n")
        f.write(f"# threshold: {args.threshold}\n")
        f.write(f"# device: {device} ({dev_name})\n")
        f.write(f"# load_s: {t_load:.3f}\n")
        f.write(f"# infer_ms: {t_infer * 1000:.1f}\n")
        f.write(f"# raw_candidates: {n_total}\n")
        f.write(f"# kept: {n_kept}\n")
        # Margin fields (blank if --target-xy not given) so a notebook can build
        # a prompt-sweep table straight from these dumps.
        if tgt is not None:
            f.write(f"# target_xy: {tgt[0]:.0f},{tgt[1]:.0f}\n")
            f.write(f"# target_tol: {args.target_tol:.0f}\n")
            f.write(f"# best_on_target: "
                    f"{best_on[0]:.4f}\n" if best_on is not None
                    else "# best_on_target: none\n")
            f.write(f"# best_off_target: "
                    f"{best_off[0]:.4f}\n" if best_off is not None
                    else "# best_off_target: none\n")
            if best_on is not None and best_off is not None:
                f.write(f"# margin: {best_on[0] - best_off[0]:+.4f}\n")
            else:
                f.write("# margin: n/a\n")
        f.write("# columns: score prompt x0 y0 x1 y1\n")
        for i in range(n_kept):
            x0, y0, x1, y1 = boxes[i]
            f.write(f"{scores[i]:.4f}\t{prompts[labels[i]]}\t"
                    f"{x0:.1f}\t{y0:.1f}\t{x1:.1f}\t{y1:.1f}\n")
    print(f"[info] detections dump -> {dump_path}")

    print("\n[done] eyeball the annotated PNG: is the red cube boxed, is the "
          "box tight, any false boxes? That read is the first finding.")


if __name__ == "__main__":
    main()

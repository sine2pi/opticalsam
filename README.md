Augments SAM3's deep memory bank with RAFT optical flow as an explicit
spatial prior. Works by injecting RAFT-warped mask prompts into SAM3's
per_frame_geometric_prompt slots, so the tracker sees BOTH its own
temporal memory AND our geometric motion prediction each frame.

"""
DEPRECATED: this module has been removed.

The instance-finding stage was replaced by:

    - prototype_proposing_loop/run_loop.py
        runs LLM-supplied select_segments() functions over the segment bank
        and writes final_prototypes_<timestamp>.json (each prototype carries
        its own matched_indices in the evaluation block).

    - LLM_prototype/prototype_initializer.py
        applies Mahalanobis-to-centroid typicality on each prototype's
        matching set and produces the (μ, p, anchor_mask) init tensors.

Importing from this file is a bug. If you find a reference, swap it for one
of the components listed above.
"""

raise ImportError(
    "LLM_prototype.instance_finder has been removed. "
    "Use prototype_proposing_loop.run_loop + LLM_prototype.prototype_initializer."
)

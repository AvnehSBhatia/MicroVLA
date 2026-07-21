"""Training scaffold for the MicroVLA trainable heads.

Contains implemented losses and data plumbing for the planner behavior-cloning
path (fusion -> TRM slot -> planner). The TRM training loss is documented but
deliberately NOT implemented here — see ``train.losses.trm_loss_documentation``
and ``microvla/trm/TRM_SPEC.md``.
"""

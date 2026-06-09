"""
DEPRECATED — moved to ``CBM/concept_bottleneck.py``.

This file's old name collided with the project-root ``model/`` package:
when Python loaded ``CBM.model``, the relative-import resolver inside it
treated subsequent ``from model.embedding ...`` lookups as
``from CBM.model.embedding ...`` and raised
``ModuleNotFoundError: ... 'model' is not a package``.

The class now lives at ``CBM/concept_bottleneck.py``. Import it as

    from CBM.concept_bottleneck import ConceptBottleneckModel

Crucially, this file does NOT re-export the class — doing so would
re-trigger the namespace collision. If you have code that still imports
``from CBM.model import ConceptBottleneckModel``, update it to use the
new path.
"""

raise ImportError(
    "CBM.model has been renamed to CBM.concept_bottleneck to avoid a "
    "namespace collision with the project-root model/ package. "
    "Update your import to: from CBM.concept_bottleneck import ConceptBottleneckModel"
)

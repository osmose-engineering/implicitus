from ai_adapter.schema.implicitus_pb2 import Model
from google.protobuf.json_format import ParseDict, MessageToDict, ParseError
from google.protobuf.message import DecodeError
import logging
import reprlib
import re
from typing import Any

class ValidationError(Exception):
    """Raised when the JSON spec cannot be parsed into the protobuf schema."""
    pass


def _validate_seed_points(obj: Any) -> None:
    """Recursively ensure seed points are unique and within bounds.

    Parameters
    ----------
    obj:
        Portion of the spec dictionary to inspect.

    Raises
    ------
    ValidationError
        If duplicate or out-of-bounds seed points are found.
    """

    if isinstance(obj, dict):
        seeds = obj.get("seed_points")
        if isinstance(seeds, list):
            # Check for duplicates
            pts = [tuple(pt) for pt in seeds]
            if len(pts) != len(set(pts)):
                raise ValidationError("Duplicate seed point detected")

            bbox_min = obj.get("bbox_min")
            bbox_max = obj.get("bbox_max")
            if bbox_min is not None and bbox_max is not None:
                for pt in seeds:
                    for p, mn, mx in zip(pt, bbox_min, bbox_max):
                        if p < mn or p > mx:
                            raise ValidationError(
                                f"Seed point {pt} outside of bounding box"  # noqa: E501
                            )
        for v in obj.values():
            _validate_seed_points(v)
    elif isinstance(obj, list):
        for item in obj:
            _validate_seed_points(item)


def _validate_edge_indices(obj: Any) -> None:
    """Recursively ensure all edges reference valid vertex indices.

    Parameters
    ----------
    obj:
        Portion of the spec dictionary to inspect.

    Raises
    ------
    ValidationError
        If any edge index is outside the bounds of ``cell_vertices``.
    """

    if isinstance(obj, dict):
        edges = obj.get("edge_list")
        verts = obj.get("cell_vertices")
        if isinstance(edges, list) and isinstance(verts, list):
            n = len(verts)
            for edge in edges:
                if not isinstance(edge, (list, tuple)):
                    continue
                for idx in edge:
                    if not isinstance(idx, int) or idx < 0 or idx >= n:
                        raise ValidationError(
                            f"Edge index {idx} out of bounds for cell_vertices of length {n}"
                        )
        for v in obj.values():
            _validate_edge_indices(v)
    elif isinstance(obj, list):
        for item in obj:
            _validate_edge_indices(item)

def validate_model_spec(spec_dict: dict, ignore_unknown_fields: bool = False) -> Model:
    """Validate and convert a raw JSON spec dictionary into a protobuf ``Model``.

    Parameters
    ----------
    spec_dict:
        Dictionary representation of the model.
    ignore_unknown_fields:
        When ``True``, unknown keys in ``spec_dict`` will be ignored rather than
        raising an error. This is useful for round-tripping models that contain
        auxiliary data (e.g., precomputed lattice fields) which are not part of
        the core schema.

    Returns
    -------
    Model
        A populated ``implicitus_pb2.Model`` instance.

    Raises
    ------
    ValidationError
        If ``spec_dict`` contains mixed-case keys or fails protobuf parsing.
    """

    def _ensure_snake_case(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if re.search(r"[A-Z]", k):
                    raise ValidationError(f"Mixed-case key detected: {k}")
                _ensure_snake_case(v)
        elif isinstance(obj, list):
            for item in obj:
                _ensure_snake_case(item)

    def _normalize_modifiers(obj):
        """Convert ``modifiers`` dictionaries into single-item lists.

        Some callers provide a mapping for ``modifiers`` even though the schema
        defines it as a repeated field. Wrapping these mappings in a list allows
        ``ParseDict`` to succeed without the caller having to perform the
        normalization themselves.
        """

        if isinstance(obj, dict):
            mods = obj.get("modifiers")
            if isinstance(mods, dict):
                obj["modifiers"] = [mods]
                mods = obj["modifiers"]
            for v in obj.values():
                _normalize_modifiers(v)
        elif isinstance(obj, list):
            for item in obj:
                _normalize_modifiers(item)

    _normalize_modifiers(spec_dict)
    _ensure_snake_case(spec_dict)
    _validate_seed_points(spec_dict)
    _validate_edge_indices(spec_dict)

    logging.debug("Normalized spec: %s", reprlib.repr(spec_dict))

    model = Model()
    try:
        # ``ParseDict`` performs field-level validation. We allow callers to
        # opt-in to ignoring unknown fields so that auxiliary data can be
        # stripped while preserving the valid portion of the model.
        ParseDict(spec_dict, model, ignore_unknown_fields=ignore_unknown_fields)
    except (TypeError, DecodeError, ValueError, ParseError) as e:
        raise ValidationError(f"Failed to validate model spec: {e}")
    return model

from ai_adapter.schema.implicitus_pb2 import Model
from ai_adapter.schema import implicitus_pb2 as pb
from google.protobuf.json_format import ParseDict, MessageToDict, ParseError
from google.protobuf.message import DecodeError
from google.protobuf.descriptor import FieldDescriptor, Descriptor
import logging
import reprlib
import re
from typing import Any

# Supported versions of the model spec. Update as new versions are introduced.
SUPPORTED_VERSIONS = {1}

class ValidationError(Exception):
    """Raised when the JSON spec cannot be parsed into the protobuf schema."""
    pass


_EXTRA_MESSAGE_DESCRIPTORS: dict[str, Descriptor] = {
    # ``booleanOp`` appears on ``Node`` even though the protobuf schema only
    # defines boolean operations as modifiers. Treat the value as a BooleanOp
    # message so its repeated fields (e.g., ``nodes``) are populated.
    "booleanOp": pb.BooleanOp.DESCRIPTOR,
    # ``lattice`` is used as a primitive wrapper for infill parameters and maps
    # to the ``Infill`` message descriptor.
    "lattice": pb.Infill.DESCRIPTOR,
    # Support explicit Voronoi lattice modifiers.
    "voronoi_lattice": pb.VoronoiLattice.DESCRIPTOR,
    "voronoiLattice": pb.VoronoiLattice.DESCRIPTOR,
}

def ensure_repeated_fields(
    obj: Any, descriptor: Descriptor = pb.Model.DESCRIPTOR
) -> None:
    """Recursively insert empty lists for missing repeated protobuf fields."""

    if not isinstance(obj, dict):
        return

    for field in descriptor.fields:
        if field.label == FieldDescriptor.LABEL_REPEATED:
            obj.setdefault(field.name, [])
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            if field.label == FieldDescriptor.LABEL_REPEATED:
                for item in obj.get(field.name, []):
                    if isinstance(item, dict):
                        ensure_repeated_fields(item, field.message_type)
            else:
                child = obj.get(field.name)
                if isinstance(child, dict):
                    ensure_repeated_fields(child, field.message_type)

    # Handle ad-hoc fields that correspond to known message types but are not
    # declared in ``descriptor`` (e.g., ``booleanOp`` on ``Node``).
    for key, value in obj.items():
        desc = _EXTRA_MESSAGE_DESCRIPTORS.get(key)
        if desc is None:
            continue
        if isinstance(value, dict):
            ensure_repeated_fields(value, desc)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    ensure_repeated_fields(item, desc)



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
            seen = set()
            degree = [0] * n
            for edge in edges:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    raise ValidationError("Edges must be 2-tuples of vertex indices")
                i, j = edge
                if not all(isinstance(idx, int) for idx in (i, j)):
                    raise ValidationError("Edge indices must be integers")
                for idx in (i, j):
                    if idx < 0 or idx >= n:
                        raise ValidationError(
                            f"Edge index {idx} out of bounds for cell_vertices of length {n}"
                        )
                if i == j:
                    raise ValidationError("Edge cannot reference the same vertex twice")
                key = (i, j) if i < j else (j, i)
                if key in seen:
                    raise ValidationError(f"Duplicate edge detected: {key}")
                seen.add(key)
                degree[i] += 1
                degree[j] += 1
            for idx, cnt in enumerate(degree):
                if cnt not in (0, 2):
                    raise ValidationError(
                        f"Vertex {idx} participates in {cnt} edges; expected 0 or 2 for closed loops"
                    )
        for v in obj.values():
            _validate_edge_indices(v)
    elif isinstance(obj, list):
        for item in obj:
            _validate_edge_indices(item)


def _primitive_bbox(primitive: dict) -> tuple[list[float], list[float]] | tuple[None, None]:
    """Return the bounding box for a primitive dict if known.

    Parameters
    ----------
    primitive:
        Mapping containing a single primitive shape specification.

    Returns
    -------
    tuple[list[float], list[float]] | tuple[None, None]
        ``(bbox_min, bbox_max)`` in object space or ``(None, None)`` if the
        primitive type is unsupported.
    """

    if not isinstance(primitive, dict):
        return None, None

    if "sphere" in primitive:
        r = primitive["sphere"].get("radius")
        if r is not None:
            return [-r, -r, -r], [r, r, r]

    if "box" in primitive:
        size = primitive["box"].get("size")
        if isinstance(size, dict):
            hx = size.get("x", 0) / 2
            hy = size.get("y", 0) / 2
            hz = size.get("z", 0) / 2
            return [-hx, -hy, -hz], [hx, hy, hz]

    if "cube" in primitive:
        s = primitive["cube"].get("size")
        if s is not None:
            half = s / 2
            return [-half, -half, -half], [half, half, half]

    if "cylinder" in primitive:
        cyl = primitive["cylinder"]
        r = cyl.get("radius")
        h = cyl.get("height")
        if r is not None and h is not None:
            return [-r, -r, 0.0], [r, r, h]

    return None, None


def _validate_bbox_contains_primitive(obj: Any) -> None:
    """Recursively ensure bbox_min/bbox_max enclose referenced primitives."""

    if isinstance(obj, dict):
        primitive = obj.get("primitive")

        # Case where bbox is specified alongside primitive
        bbox_min = obj.get("bbox_min")
        bbox_max = obj.get("bbox_max")
        if primitive and bbox_min is not None and bbox_max is not None:
            pmin, pmax = _primitive_bbox(primitive)
            if pmin and pmax:
                for mn, pmn in zip(bbox_min, pmin):
                    if mn > pmn:
                        raise ValidationError("bbox_min does not enclose primitive")
                for mx, pmx in zip(bbox_max, pmax):
                    if mx < pmx:
                        raise ValidationError("bbox_max does not enclose primitive")

        # Case where bbox resides under an infill modifier
        modifiers = obj.get("modifiers")
        if primitive and modifiers is not None:
            if isinstance(modifiers, dict):
                mods_iter = [modifiers]
            elif isinstance(modifiers, list):
                mods_iter = [m for m in modifiers if isinstance(m, dict)]
            else:
                mods_iter = []
            for mod in mods_iter:
                infill = mod.get("infill")
                if isinstance(infill, dict):
                    bmin = infill.get("bbox_min")
                    bmax = infill.get("bbox_max")
                    if bmin is not None and bmax is not None:
                        pmin, pmax = _primitive_bbox(primitive)
                        if pmin and pmax:
                            for mn, pmn in zip(bmin, pmin):
                                if mn > pmn:
                                    raise ValidationError(
                                        "bbox_min does not enclose primitive"
                                    )
                            for mx, pmx in zip(bmax, pmax):
                                if mx < pmx:
                                    raise ValidationError(
                                        "bbox_max does not enclose primitive"
                                    )

        for v in obj.values():
            _validate_bbox_contains_primitive(v)

    elif isinstance(obj, list):
        for item in obj:
            _validate_bbox_contains_primitive(item)

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

    # Ensure the spec declares a supported version. A copy is made so that the
    # version field can be stripped before protobuf parsing.
    version = spec_dict.get("version")
    if version is None:
        raise ValidationError("Missing version field")
    if version not in SUPPORTED_VERSIONS:
        raise ValidationError(f"Unrecognized version: {version}")
    spec_dict = dict(spec_dict)
    spec_dict.pop("version", None)

    _normalize_modifiers(spec_dict)
    _ensure_snake_case(spec_dict)
    _validate_seed_points(spec_dict)
    _validate_edge_indices(spec_dict)
    _validate_bbox_contains_primitive(spec_dict)

    def _strip_constraints(obj: Any) -> None:
        """Remove ``constraints`` keys until protobuf definitions support them.

        The current protobuf schema used for validation does not define a
        ``constraints`` field on ``Node``. Downstream components expect models to
        include this key (defaulting to an empty list), so the validation step
        temporarily strips it to avoid parse errors. Once the schema is updated,
        this helper can be removed.
        """

        if isinstance(obj, dict):
            obj.pop("constraints", None)
            for v in obj.values():
                _strip_constraints(v)
        elif isinstance(obj, list):
            for item in obj:
                _strip_constraints(item)

    _strip_constraints(spec_dict)

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

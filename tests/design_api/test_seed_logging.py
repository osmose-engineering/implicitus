import logging

from design_api.services.seed_utils import resolve_seed_spec
from design_api.services.mapping import map_to_proto_dict


def test_uniform_seed_logs(caplog):
    request_id = "req-123"
    seeds = [[0.0, 0.0, 0.0]]
    cell_vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    edge_list = [[0, 1]]
    spec = {
        "shape": "sphere",
        "size_mm": 1.0,
        "modifiers": {
            "infill": {
                "mode": "uniform",
                "seed_points": seeds,
                "cell_vertices": cell_vertices,
                "edge_list": edge_list,
            }
        },
    }

    with caplog.at_level(logging.INFO, logger="design_api.services.seed_utils"):
        resolve_seed_spec({}, [0, 0, 0], [1, 1, 1], seed_points=seeds, mode="uniform", request_id=request_id)
    with caplog.at_level(logging.INFO, logger="design_api.services.mapping"):
        map_to_proto_dict(spec, request_id=request_id)

    seed_logs = [
        r
        for r in caplog.records
        if getattr(r, "request_id", None) == request_id and getattr(r, "seed_point", None)
    ]
    mapping_logs = [
        r
        for r in caplog.records
        if getattr(r, "request_id", None) == request_id and getattr(r, "edge_indices", None)
    ]
    assert seed_logs, "Expected seed logging entries"
    assert mapping_logs, "Expected mapping log entries with edge indices"

# ai_adapter/csg_adapter.py
from ai_adapter.schema.implicitus_pb2 import Model

def parse_and_build_model(llm_output) -> Model:
    proto = Model()
    proto.ParseFromString(llm_output)
    return Model.from_proto(proto)
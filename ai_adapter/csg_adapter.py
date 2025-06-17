# ai_adapter/csg_adapter.py
# ai_adapter/csg_adapter.py
import json
import base64
from google.protobuf.json_format import ParseDict
from ai_adapter.schema.implicitus_pb2 import Model

def parse_and_build_model(llm_output) -> Model:
    proto = Model()
    # unwrap lists
    if isinstance(llm_output, list) and len(llm_output) > 0:
        llm_output = llm_output[0]

    # handle dict (already parsed JSON)
    if isinstance(llm_output, dict):
        ParseDict(llm_output, proto)

    # handle JSON or base64-encoded strings or raw bytes-as-text
    elif isinstance(llm_output, str):
        # try JSON
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError:
            # try base64
            try:
                decoded = base64.b64decode(llm_output)
                proto.ParseFromString(decoded)
            except Exception:
                # fallback to raw utf-8 bytes
                proto.ParseFromString(llm_output.encode('utf-8'))
        else:
            # it was JSON
            ParseDict(data, proto)

    # handle bytes/bytearray
    elif isinstance(llm_output, (bytes, bytearray)):
        proto.ParseFromString(llm_output)

    else:
        raise TypeError(f"Unsupported llm_output type: {type(llm_output)}")

    return Model.from_proto(proto)
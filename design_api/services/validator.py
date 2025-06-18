from ai_adapter.schema.implicitus_pb2 import Model
from google.protobuf.json_format import ParseDict, MessageToDict
from google.protobuf.message import DecodeError

class ValidationError(Exception):
    """Raised when the JSON spec cannot be parsed into the protobuf schema."""
    pass

def validate_model_spec(spec_dict: dict) -> Model:
    """
    Validate and convert a raw JSON spec dictionary into a protobuf Model.

    Args:
        spec_dict: A dictionary parsed from LLM JSON output.

    Returns:
        A populated implicitus_pb2.Model instance.

    Raises:
        ValidationError: If the JSON does not match the protobuf schema.
    """
    model = Model()
    try:
        # ParseDict will perform field-level validation
        ParseDict(spec_dict, model, ignore_unknown_fields=False)
    except (TypeError, DecodeError, ValueError) as e:
        raise ValidationError(f"Failed to validate model spec: {e}")
    return model

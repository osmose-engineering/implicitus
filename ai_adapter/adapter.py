# ai_adapter/adapter.py
from ai_adapter.schema import implicitus_pb2

def generate_model(prompt: str) -> implicitus_pb2.Model:
    """
    Stub: build a simple implicit model for the given prompt.
    Currently returns a unit sphere with the prompt as its ID.
    """
    model = implicitus_pb2.Model()
    model.id = prompt

    # Create a root node containing a Sphere primitive
    root = implicitus_pb2.Node()
    primitive = implicitus_pb2.Primitive()
    sphere = implicitus_pb2.Sphere(radius=1.0)
    primitive.sphere.CopyFrom(sphere)
    root.primitive.CopyFrom(primitive)

    model.root.CopyFrom(root)
    return model
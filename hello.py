import modal

# Create a Modal App
stub = modal.App("hello_world_app")

# Build an image with FastAPI installed
image = (
    modal.Image.debian_slim()
    .pip_install(["fastapi"])  # <- you MUST include fastapi
)

# Define a function and expose it as a FastAPI web endpoint
@stub.function(image=image)
@modal.fastapi_endpoint()
def hello():
    return "Hello, world!"

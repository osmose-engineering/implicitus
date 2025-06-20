from setuptools import setup, find_packages

setup(
    name="implicitus",
    version="0.1",
    packages=find_packages(),  # automatically include ai_adapter, design_api, etc.
    include_package_data=True,
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.22.0",
        "transformers>=4.0.0",
        "protobuf",
        # add any other runtime dependencies here
    ],
    extras_require={
        "dev": ["pytest", "httpx"],  # for testing
    },
)

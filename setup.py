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
        "python-dotenv>=1.1.1",
        "SQLAlchemy>=1.4.0",
        "alembic>=1.16.4",
        "asyncpg>=0.30.0",
        "redis>=4.5.0",
        "sentencepiece>=0.2.0",
        "huggingface-hub>=0.34.1",
        "openai>=1.97.1",
    ],
    extras_require={
        "dev": ["pytest", "httpx"],  # for testing
    },
)

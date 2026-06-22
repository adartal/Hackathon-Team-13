from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "local-s3-access-key"
    s3_secret_key: str = "local-s3-secret-key"
    s3_region: str = "us-east-1"
    s3_default_bucket: str = "math-tutor-assets"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

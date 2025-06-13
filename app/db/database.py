import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CA_CERT = os.getenv("CA_CERT")  # Full certificate content (multiline or \n)

def create_database_engine():
    """Create database engine with proper SSL configuration for Aiven MySQL"""
    
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Check if we need SSL configuration (for Aiven MySQL)
    if CA_CERT and CA_CERT.strip():
        # Write CA cert to a temporary file
        ca_cert_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pem", mode='w')
        ca_cert_file.write(CA_CERT.replace('\\n', '\n'))  # Handle escaped newlines
        ca_cert_file.close()
        
        # Connect with SQLAlchemy using temp CA cert
        connect_args = {
            "ssl": {
                "ca": ca_cert_file.name
            }
        }
        
        engine = create_engine(
            DATABASE_URL,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False  # Set to True for SQL debugging
        )
    else:
        # For local development or non-SSL connections
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
    
    return engine

# Create the engine
engine = create_database_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_database_connection():
    """Test database connection"""
    try:
        from sqlalchemy import text
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

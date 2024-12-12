from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from main import User  # Ensure you import your User model correctly

# Create the SQLite engine and point it to test.db
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)

# Create a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

# Create a new user
new_user = User(
    id="user1",  # Example ID, this can be any unique string
    email="user1@example.com",  # Example email
    first_name="John",  # User's first name
    last_name="Doe",  # User's last name
    mobile_number="+1234567890",  # Example mobile number with country code
    address="123 Main St",  # Example address
    city="New York",  # Example city
    state="NY",  # Example state
    zip_postal="10001",  # Example zip/postal code
    country="USA",  # Example country
    role="user",  # Default role
    sweeps_points=1000.0,  # Default starting balance
    betting_points=1000.0,  # Default betting points
    ban=False,  # User is not banned
    created_at=datetime.utcnow()  # The current UTC time
)

# Add the new user to the session and commit the changes
db.add(new_user)
db.commit()

# Refresh the session to get the updated object (including auto-generated fields)
db.refresh(new_user)

# Print the inserted user data
print(new_user.as_dict())

# Close the session
db.close()

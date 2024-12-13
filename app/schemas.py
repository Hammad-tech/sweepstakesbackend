from pydantic import BaseModel, EmailStr, constr, root_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum as PyEnum
from .models import UserRole, RemarkType


class MatchCreate(BaseModel):
    home_team: str
    away_team: str
    match_time: str
    variations: Optional[List[dict]] = []  # Optional JSON field for variations


# Define a Pydantic model for the response
class MatchResponse(BaseModel):
    id: int
    home_team: str
    away_team: str
    match_time: str
    total_yes_bets: int
    total_no_bets: int
    variations: List[dict] = []

    class Config:
        orm_mode = True


# Pydantic model for creating a new user
class UserCreate(BaseModel):
    email: EmailStr
    password: constr(min_length=8)  # Password must be at least 8 characters
    name: Optional[str] = None
    country: Optional[str] = None
    role: UserRole = UserRole.USER  # Default role


# Pydantic model for returning user data
class UserOut(BaseModel):
    id: int
    email: EmailStr
    name: Optional[str]
    country: Optional[str]
    created_at: datetime
    role: UserRole

    class Config:
        orm_mode = True  # Enable ORM mode
        arbitrary_types_allowed = True


class EventResponse(BaseModel):
    id: int
    match_id: int
    match_time: datetime  # Include match_time from the related Match table
    question: str
    total_yes_bets: int
    total_no_bets: int
    yes_percentage: int
    variations: List[dict]  # You can adjust this if variations is a more complex type

    class Config:
        orm_mode = True  # This allows FastAPI to automatically convert SQLAlchemy models to Pydantic models


# CreateRemark model that is used to validate the incoming data
class CreateRemark(BaseModel):
    user_id: str
    amount: float
    message: str
    type: RemarkType


class MatchBetUpdate(BaseModel):
    match_id: int
    bet: int


# Pydantic model for returning match data
class MatchOut(BaseModel):
    id: int
    home_team: str
    away_team: str
    match_time: str
    total_yes_bets: int
    total_no_bets: int

    class Config:
        orm_mode = True  # This tells Pydantic to use ORM objects


class UserLogin(BaseModel):
    email: EmailStr  # Email as EmailStr for validation
    password: str  # Password for login

    class Config:
        orm_mode = True  # To allow conversion to and from ORM models


class ShareOut(BaseModel):
    id: int
    user_id: int
    bet_id: int
    amount: float
    created_at: datetime

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True


class RegisterUser(BaseModel):
    name: Optional[str] = None  # Name is optional in case it's not provided
    email: EmailStr  # Use EmailStr for email validation
    password: str  # Password field
    first_name: Optional[str] = None  # First name, optional
    last_name: Optional[str] = None  # Last name, optional
    mobile_number: Optional[str] = None  # Mobile number, optional
    address: Optional[str] = None  # Address, optional
    city: Optional[str] = None  # City, optional
    state: Optional[str] = None  # State, optional
    zip_postal: Optional[str] = None  # Zip code, optional
    country: Optional[str] = None  # Country, optional
    role: Optional[str] = "USER"  # Default role for new users

    class Config:
        orm_mode = True  # To allow conversion to and from ORM models


class BuyShareRequest(BaseModel):
    user_id: str
    event_id: int
    outcome: int
    shareCount: int
    share_price: float


class SellShareRequest(BaseModel):
    user_id: str
    eventId: int
    outcome: int
    shareCount: int
    share_price: float


class UserProfile(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    mobile_number: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_postal: Optional[str] = None
    country: Optional[str] = None
    role: str
    sweeps_points: float
    betting_points: float
    ban: bool
    created_at: str  # ISO format string for created_at

    @root_validator(pre=True)
    def format_datetime(cls, values):
        if "created_at" in values and isinstance(values["created_at"], datetime):
            values["created_at"] = values[
                "created_at"
            ].isoformat()  # Convert to ISO string format
        return values

    class Config:
        orm_mode = True


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


# Define the request model for updating profile
class UserProfileEdit(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    mobile_number: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_postal: Optional[str] = None
    country: Optional[str] = None
    add_balance: Optional[float] = None
    subtract_balance: Optional[float] = None
    ban: Optional[bool] = None

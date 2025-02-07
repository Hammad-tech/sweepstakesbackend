from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    JSON,
    func,
    Enum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.ext.mutable import MutableList
from enum import Enum as PyEnum
from typing import List, Optional
from datetime import datetime, timedelta
import time as t
from .auth import (
    hash_password,
    verify_password,
)  # Ensure hash and verify are imported correctly
from .db_config import Base


class UserRole(str, PyEnum):
    ADMIN = "admin"
    USER = "user"


# Define the RemarkType Enum correctly
class RemarkType(PyEnum):
    addbalance = "addBalance"
    subbalance = "subBalance"
    ban = "ban"

class SportType(PyEnum):
    basketball = "basketball"
    football = "football"
    hockey = "hockey"

class MarketType(PyEnum):
    POINTS = "points"
    OUTCOME = "outcome"

class EventType(str, PyEnum):
    MATCH_RESULT = "match_result"  # Which team will win
    TOTAL_POINTS = "total_points"  # Combined points of both teams
    TEAM_POINTS = "team_points"  # Points for a specific team
    OVER_UNDER = "over_under"  # Over/under events for total points
    HANDICAP = "handicap"


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    type = Column(String, nullable=False)  # e.g., MATCH_RESULT, OVER_UNDER
    heading = Column(String, nullable=False)  # e.g., "Match Result - 100 Indices"
    question = Column(String, nullable=False)
    threshold = Column(Float, nullable=True)  # For Over/Under markets
    buy_sell_index = Column(Float, default=100.0)  # Initial index value
    buy_price = Column(Float, default=51.0)  # Starting buy price
    sell_price = Column(Float, default=49.0)  # Starting sell price (2-point spread)
    variations = Column(JSON, default=[])  # List of buy/sell price changes over time

    resolved = Column(Boolean, default=False)  # ✅ Ensure this exists

    match = relationship("Match", back_populates="events")
    shares = relationship("Share", back_populates="event") 



class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    sport = Column(Enum(SportType), nullable=False)  # Sport Type (Basketball, Football, etc.)
    league = Column(String, nullable=False)
    team1 = Column(String, nullable=False)
    team2 = Column(String, nullable=False)
    match_time = Column(DateTime, nullable=False)
    bet_end_time = Column(DateTime)

    status = Column(String, default="scheduled")  # Stores match status
    result = Column(JSON, default={})  # Stores quarter-wise scores, total points, and winner dynamically
    total_points = Column(Integer, default=0)  # Stores sum of all quarters

    events = relationship("Event", back_populates="match", cascade="all, delete-orphan")

    def as_dict(self):
        return {
            column.name: (
                getattr(self, column.name).isoformat()
                if isinstance(getattr(self, column.name), datetime)
                else getattr(self, column.name)
            )
            for column in self.__table__.columns
        }


class Share(Base):
    __tablename__ = "shares"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    amount = Column(Float, nullable=False)  # Number of shares
    bet_type = Column(String, nullable=False)  # "buy" or "sell"
    share_price = Column(Float, nullable=False)  # Price at which shares were bought/sold
    limit_price = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="shares")
    event = relationship("Event", back_populates="shares")


# Define the Remarks model
class Remarks(Base):

    __tablename__ = "remarks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(
        Enum(RemarkType), name="remark_type_enum", nullable=False
    )  # Use Enum(RemarkType)
    message = Column(String, nullable=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)  # Timestamp field

    # Define relationship to User
    user = relationship("User", back_populates="remarks")

    def as_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type.value,  # Return the string value of the Enum
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),  # Format timestamp to ISO 8601 string
        }


class User(Base):

    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    role = Column(String, nullable=False, default=UserRole.USER.value)
    sweeps_points = Column(
        Float, default=5000.0
    )  # Default starting balance of Sweeps Points
    betting_points = Column(
        Float, default=5000.0
    )  # Default starting balance of Betting Points
    ban = Column(Boolean, default=False)
    first_name = Column(String, nullable=True)  # User's first name
    last_name = Column(String, nullable=True)  # User's last name
    mobile_number = Column(
        String, nullable=True
    )  # User's mobile number with country code
    address = Column(String, nullable=True)  # Street address
    city = Column(String, nullable=True)  # City of residence
    state = Column(String, nullable=True)  # State or province
    zip_postal = Column(String, nullable=True)  # Zip or postal code
    country = Column(String, nullable=True)  # Country of residence
    shares = relationship("Share", back_populates="user")
    remarks = relationship(
        "Remarks", back_populates="user"
    )  # New relationship for remarks

    def as_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "mobile_number": self.mobile_number,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_postal": self.zip_postal,
            "country": self.country,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "role": self.role,
            "sweeps_points": self.sweeps_points,
            "betting_points": self.betting_points,
            "ban": self.ban,
        }

    @staticmethod
    def hash_password(password: str) -> str:
        return hash_password(password)

    @staticmethod
    def verify_password(hashed_password: str, plain_password: str) -> bool:
        return verify_password(hashed_password, plain_password)

    def update_points(self, amount: float, point_type: str, transaction_type: str):
        """
        Updates the user's points based on the share transaction.

        :param amount: Amount to add or subtract from the balance.
        :param point_type: Either "sweeps_points" or "betting_points".
        :param transaction_type: Either "buy" or "sell".
        """
        if point_type not in ["sweeps_points", "betting_points"]:
            raise ValueError(
                "Invalid point type. Must be 'sweeps_points' or 'betting_points'."
            )

        if transaction_type == "buy":
            if getattr(self, point_type) < amount:
                raise ValueError(f"Insufficient {point_type} balance.")
            setattr(self, point_type, getattr(self, point_type) - amount)
        elif transaction_type == "sell":
            setattr(self, point_type, getattr(self, point_type) + amount)
        else:
            raise ValueError("Invalid transaction type. Must be 'buy' or 'sell'.")

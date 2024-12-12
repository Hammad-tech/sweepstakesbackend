import cred as cred
import firebase_admin.exceptions
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine,Enum, Column, Integer, String, Time, Float, ForeignKey, DateTime, JSON, func, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import sessionmaker, Session, relationship
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Optional
from pydantic import BaseModel, EmailStr, constr, PositiveFloat, Field, root_validator
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import json
import time as t
from enum import Enum as PyEnum
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth import hash_password, verify_password
from bs4 import BeautifulSoup
import pytz
from typing import List, Dict
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

''' if you want to change anything in models you have to make migrations now 
first run :: alembic revision --autogenerate -m "some comment like what you have added or removed"
To apply migrations run :: alembic upgrade head

'''


# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
target_metadata = Base.metadata

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)  # Foreign key to Match table
    question = Column(String,default="")
    total_yes_bets = Column(Integer, default=0)  # Initialize to 0
    total_no_bets = Column(Integer, default=0)  # Initialize to 0
    variations = Column(MutableList.as_mutable(JSON), default=[])

    shares = relationship("Share", back_populates="event", cascade="all, delete-orphan")  # Updated to Share
    match = relationship("Match", back_populates="events")

    def as_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


# Database Models
class Match(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True, index=True)
    team1 = Column(String)
    team2 = Column(String)
    match_time = Column(DateTime)
    league = Column(String)
    bet_start_time = Column(DateTime)  # Betting start time
    bet_end_time = Column(DateTime)    # Betting end time

    events = relationship("Event", back_populates="match", cascade="all, delete-orphan")

    def as_dict(self):
        return {
            column.name: (getattr(self, column.name).isoformat() if isinstance(getattr(self, column.name), datetime) else getattr(self, column.name))
            for column in self.__table__.columns
        }
    

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


class UserRole(str, PyEnum):
    ADMIN = "admin"
    USER = "user"


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


class Share(Base):
    __tablename__ = "shares"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"))
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)  # Foreign key to Event table
    amount = Column(Float)  # Amount of the share
    bet_type = Column(String)  # "yes" or "no"
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="shares")
    event = relationship("Event", back_populates="shares")  # Updated to back-populate from Share to Event



# Define the RemarkType Enum correctly
class RemarkType(PyEnum):
    addbalance = "addBalance"
    subbalance = "subBalance"
    ban = "ban"

# CreateRemark model that is used to validate the incoming data
class CreateRemark(BaseModel):
    user_id: str
    amount: float
    message: str
    type: RemarkType
    
# Define the Remarks model
class Remarks(Base):
    __tablename__ = 'remarks'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    type = Column(Enum(RemarkType), name="remark_type_enum", nullable=False)  # Use Enum(RemarkType)
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
            "timestamp": self.timestamp.isoformat()  # Format timestamp to ISO 8601 string
        }



class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    role = Column(String, nullable=False, default=UserRole.USER.value)
    sweeps_points = Column(Float, default=1000.0)  # Default starting balance of Sweeps Points
    betting_points = Column(Float, default=1000.0)  # Default starting balance of Betting Points
    ban = Column(Boolean, default=False)
    first_name = Column(String, nullable=True)  # User's first name
    last_name = Column(String, nullable=True)  # User's last name
    mobile_number = Column(String, nullable=True)  # User's mobile number with country code
    address = Column(String, nullable=True)  # Street address
    city = Column(String, nullable=True)  # City of residence
    state = Column(String, nullable=True)  # State or province
    zip_postal = Column(String, nullable=True)  # Zip or postal code
    country = Column(String, nullable=True)  # Country of residence
    shares = relationship("Share", back_populates="user")
    remarks = relationship("Remarks", back_populates="user")  # New relationship for remarks

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
            raise ValueError("Invalid point type. Must be 'sweeps_points' or 'betting_points'.")

        if transaction_type == "buy":
            if getattr(self, point_type) < amount:
                raise ValueError(f"Insufficient {point_type} balance.")
            setattr(self, point_type, getattr(self, point_type) - amount)
        elif transaction_type == "sell":
            setattr(self, point_type, getattr(self, point_type) + amount)
        else:
            raise ValueError("Invalid transaction type. Must be 'buy' or 'sell'.")

   

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


# Create the tables
Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
        if 'created_at' in values and isinstance(values['created_at'], datetime):
            values['created_at'] = values['created_at'].isoformat()  # Convert to ISO string format
        return values

    class Config:
        orm_mode = True

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str




# Web Scraper Function with Database Insertion

@app.get("/api/scrape_and_store_matches")
def scrape_and_store_matches(db: Session = Depends(get_db)):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # URL to scrape
    url = "https://www.pinnacle.com/en/basketball/matchups/"

    # Open the URL
    driver.get(url)

    # Wait for the page to fully load
    try:
        WebDriverWait(driver, 50).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".list-mCW1NFV2s6"))
        )
    except:
        return {"error": "Timeout waiting for the page to load."}

    matches_list: List[Match] = []

    # Find the scrollable table container
    scrollable_table = driver.find_element(By.CSS_SELECTOR, ".list-mCW1NFV2s6")

    # List to store scraped data
    data = []

    # Set to track matches we've already processed (by team names and match time)
    seen_matches = set()

    # Initial height of the content
    last_height = driver.execute_script(
        "return arguments[0].scrollHeight", scrollable_table
    )

    # League variable to store the current league name
    current_league = ""
    # Continuously scroll and scrape the content
    while True:
        # Get the page source after this scroll
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Find all game sections on the page
        games = soup.find_all("div", class_="scrollbar-item")
        
        # Loop through each game and extract the required information
        for game in games:
            # Extract league header using the updated class
            league_header = game.find("div", class_="row-u9F3b9WCM3 row-CTcjEjV6yK")
            if league_header:
                league_name = league_header.find("span", class_="ellipsis")
                if league_name:
                    current_league = league_name.text.strip()

            # Extract the teams' names and match time (only if there is match data)
            teams = game.find_all("div", class_="gameInfoLabel-EDDYv5xEfd")
            if len(teams) > 0:
                team_names = [team.text.strip() for team in teams]
            else:
                continue  # Skip if no valid teams data found

            match_time = game.find("div", class_="matchupDate-tnomIYorwa")
            match_time = match_time.text.strip() if match_time else "N/A"

            # Check if the match has already been processed by combining team names and match time
            match_identifier = f"{team_names[0]} vs {team_names[1]} at {match_time}"
            if match_identifier in seen_matches:
                continue  # Skip if this match has already been processed

            seen_matches.add(match_identifier)  # Mark this match as processed

            # Extract all buttons (Handicap, Money Line, Over/Under)
            buttons = game.find_all("button", title=True)

            # Extract Handicap values and corresponding Money Line prices (first two buttons)
            handicap = [{}, {}]
            if len(buttons) >= 2:  # Ensure there are at least two buttons for Handicap
                for i in range(2):
                    # Check if the span with class 'label-GT4CkXEOFj' exists
                    value_span = buttons[i].find("span", class_="label-GT4CkXEOFj")
                    price_span = buttons[i].find("span", class_="price-r5BU0ynJha")
                    if value_span and price_span:
                        value = value_span.text.strip()
                        price = price_span.text.strip()
                        handicap[i]["Value"] = value
                        handicap[i]["Price"] = price

            # Extract Over/Under values (last two buttons)
            over = {}
            under = {}
            if (
                    len(buttons) >= 4
            ):  # Ensure there are at least four buttons (two for Over/Under)
                over_value_span = buttons[2].find("span", class_="label-GT4CkXEOFj")
                over_price_span = buttons[2].find("span", class_="price-r5BU0ynJha")
                if over_value_span and over_price_span:
                    value = over_value_span.text.strip()
                    price = over_price_span.text.strip()
                    over["Value"] = value
                    over["Price"] = price

                under_value_span = buttons[3].find("span", class_="label-GT4CkXEOFj")
                under_price_span = buttons[3].find("span", class_="price-r5BU0ynJha")
                if under_value_span and under_price_span:
                    value = under_value_span.text.strip()
                    price = under_price_span.text.strip()
                    under["Value"] = value
                    under["Price"] = price

            # Extract Money Line values for both teams (from the separate section)
            money_line = []
            money_line_buttons = game.find_all("button", class_="market-btn")

            # Ensure we get both Money Line odds
            for button in money_line_buttons:
                price = button.find("span", class_="price-r5BU0ynJha")
                if price:
                    money_line.append(price.text.strip())

            if len(money_line) == 6:
                money_line = [money_line[2], money_line[3]]
            else:
                money_line = []

            # Store extracted information in a dictionary
            game_data = {
                "League": current_league,
                "Team1": team_names[0],
                "Team2": team_names[1],
                "Match Time": match_time,
            }

            data.append(game_data)

            # Parse the time string into a time object
            parsed_time = datetime.strptime(match_time, "%H:%M").time()
            today = datetime.today()
            local_time = datetime.combine(today, parsed_time)
            local_tz = pytz.timezone('UTC')  # Use UTC timezone (or adjust as needed)
            local_time = local_tz.localize(local_time)

            match = Match(
                team1=team_names[0],
                team2=team_names[1],
                match_time=local_time,
                bet_start_time= local_time - timedelta(hours=5),
                bet_end_time = local_time - timedelta(minutes=10),
                league=current_league,
            )
            matches_list.append(match)
            db.add(match)
            db.commit()
            db.refresh(match)
            # Create Event question for the match
            event = Event(
                match_id=match.id,  # Link event to match using match ID
                question=f"Will {team_names[0]} win against {team_names[1]}?",
                total_yes_bets=0,  # Initialize with 0 votes for yes
                total_no_bets=0,  # Initialize with 0 votes for no
                variations=[]  # Add any variations if needed
            )

            db.add(event)
            db.commit()

        # Scroll the table down incrementally
        driver.execute_script(
            "arguments[0].scrollTop += arguments[0].offsetHeight", scrollable_table
        )
        t.sleep(2)  # Allow time for the content to load

        # Get the new height after scrolling
        new_height = driver.execute_script(
            "return arguments[0].scrollHeight", scrollable_table
        )

        # If the height hasn't changed, we have reached the bottom
        if new_height == last_height:
            break

        last_height = new_height

    # Close the browser once data is scraped
    driver.quit()

    return {"message": f"{len(matches_list)} matches scraped and stored successfully!"}


@app.post("/create_test_match")
def create_match(match: MatchCreate, db: Session = Depends(get_db)):
    # Create a new match object with the received data
    db_match = Match(
        home_team=match.home_team,
        away_team=match.away_team,
        match_time=match.match_time,
        total_yes_bets=0,  # Default to 0 for a new match
        total_no_bets=0,  # Default to 0 for a new match
    )

    # Add the match to the session and commit
    db.add(db_match)
    db.commit()
    db.refresh(db_match)  # Refresh to get the ID and other auto-generated fields from the database

    return {"message": "Test match created successfully", "match": db_match.as_dict()}


# API to retrieve all matches
@app.get("/matches/")
def get_matches(db: Session = Depends(get_db)):
    matches = db.query(Match).all()
    matches_list = [match.as_dict() for match in matches]
    return JSONResponse(content=matches_list)


@app.get("/event/{event_id}")
def get_event_by_id(event_id: int, db: Session = Depends(get_db)):
    # Query the database for the event with the given ID
    db_event = db.query(Event).filter(Event.id == event_id).first()

    if not db_event:
        raise HTTPException(status_code=404, detail=f"Event with ID {event_id} not found")

    # Query the associated match for this event
    db_match = db.query(Match).filter(Match.id == db_event.match_id).first()

    # If no match is found, raise an error
    if not db_match:
        raise HTTPException(status_code=404, detail=f"Match for event ID {event_id} not found")
    # Initialize variables to calculate percentages
    total_yes_bets = db_event.total_yes_bets
    total_no_bets = db_event.total_no_bets

    # Calculate yes/no percentages based on the current state
    if total_yes_bets + total_no_bets > 0:
        yes_percentage = (total_yes_bets / (total_yes_bets + total_no_bets)) * 100
        no_percentage = 100 - yes_percentage
    else:
        yes_percentage = 50
        no_percentage = 50
    
    if yes_percentage == 0:
        yes_percentage = 1
        no_percentage = 99

    elif no_percentage == 0:
        yes_percentage = 99
        no_percentage = 1

    # Create a new variation entry with the current timestamp and percentages
    new_variation = {
        "timestamp": str(func.now()),  # Current time
        "yes": round(yes_percentage, 2),
        "no": round(no_percentage, 2)
    }

    # Append the new variation to the event's variations list
    db_event.variations.append(new_variation)

    # Commit the changes to the database
    db.commit()

    # Return the event details along with the match and variations
    response_data = {
        "id": db_event.id,
        "match_id": db_event.match_id,
        "question": db_event.question,
        "total_yes_bets": db_event.total_yes_bets,
        "total_no_bets": db_event.total_no_bets,
        "variations": db_event.variations,
        "match": {
            "id": db_match.id,
            "team1": db_match.team1,
            "team2": db_match.team2,
            "match_time": str(db_match.match_time),
            "league": db_match.league,
            "bet_start_time": str(db_match.bet_start_time),
            "bet_end_time": str(db_match.bet_end_time)
        }
    }

    return JSONResponse(content=response_data)


@app.get("/events", response_model=List[EventResponse])  # Return a list of EventResponse objects
def get_events(db: Session = Depends(get_db)):
 
    current_time = func.now()

    # Query the matches that have a bet_end_time greater than the current time
    matches = db.query(Match).filter(Match.bet_end_time > current_time).all()
    # If no matches, return an empty list instead of raising an error
    events = []
    for match in matches:
        # Get all events for this match
        match_events = db.query(Event).filter(Event.match_id == match.id).all()

        # Add the events to the list, including the match_time from the related Match table
        for event in match_events:
            event_data = event.as_dict()
            event_data["match_time"] = match.match_time  # Include match_time from the Match table
            # Initialize variables to calculate percentages
            total_yes_bets = event.total_yes_bets
            total_no_bets = event.total_no_bets

            # Calculate yes/no percentages based on the current state
            if total_yes_bets + total_no_bets > 0:
                yes_percentage = (total_yes_bets / (total_yes_bets + total_no_bets)) * 100
            else:
                yes_percentage = 50

            event_data["yes_percentage"] = yes_percentage
            events.append(event_data)

    return events  # FastAPI will automatically serialize the events using the EventResponse Pydantic model

@app.post("/modifyBalance", response_model=dict)
def modify_balance(create_remark: CreateRemark, db: Session = Depends(get_db)):
    # Check if the user exists
    user = db.query(User).filter(User.id == create_remark.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create the Remark
    new_remark = Remarks(
        user_id=create_remark.user_id,
        type=create_remark.type,
        message=create_remark.message,
    )
    
    # Update user's balance based on the type (addBalance or subBalance)
    if create_remark.type == RemarkType.addbalance:
        user.sweeps_points += create_remark.amount
    elif create_remark.type == RemarkType.subbalance:
        if user.sweeps_points < create_remark.amount:
            raise HTTPException(status_code=400, detail="Insufficient balance to subtract")
        user.sweeps_points -= create_remark.amount
    
    # Add the new remark and update the user
    db.add(new_remark)
    db.commit()
    
    # Return response
    return {"message": "Remark added successfully", "user_id": create_remark.user_id, "amount": create_remark.amount}

@app.post("/ban_unban")
def ban_unban_user(user_id: str, message: str, db: Session = Depends(get_db)):
    # Fetch the user from the database
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    
    user.ban = not(user.ban)

    # Create a new remark in the remarks table
    remark = Remarks(
        user_id=user.id,
        type=RemarkType.ban,  # Set the type as 'ban'
        message=message,
    )

    # Add the user ban and the remark to the session
    db.add(remark)
    db.commit()

    return {"message": f"User {user_id} has been banned successfully.", "user_id": user_id, "ban_status": user.ban}


@app.post("/update-bets/")
def update_bets(match_list: List[MatchBetUpdate], db: Session = Depends(get_db)):
    for match in match_list:
        # Get the match by ID (use ORM model for querying)
        db_match = db.query(Match).filter(Match.id == match.id).first()

        if not db_match:
            raise HTTPException(status_code=404, detail=f"Match with ID {match.id} not found")

        # Update yes or no bet count based on the 'bet' field
        if match.bet == 1:
            db_match.total_yes_bets += 1
            # Create a new 'yes' bet with timestamp
            new_bet = Bet(match_id=db_match.id, bet_type="yes", timestamp=func.now())
            db.add(new_bet)  # Add the new 'yes' bet to the database
        elif match.bet == 0:
            db_match.total_no_bets += 1
            # Create a new 'no' bet with timestamp
            new_bet = Bet(match_id=db_match.id, bet_type="no", timestamp=func.now())
            db.add(new_bet)  # Add the new 'no' bet to the database
        else:
            raise HTTPException(status_code=400, detail="Invalid bet value. Use 0 for no and 1 for yes.")

        # Commit changes to the database
        db.commit()
        db.refresh(db_match)

    return {"message": "Bet counts and timestamps updated successfully"}


def load_data_from_file():
    try:
        with open("test.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": str(e)}


@app.get("/bettors")
async def get_bettors():
    """Return the JSON data read from the file."""
    data = load_data_from_file()
    return JSONResponse(content=data)


@app.get("/users/", response_model=List[UserOut])  # Define the response model
def get_users(db: Session = Depends(get_db)):
    # Query all users from the database
    users = db.query(User).all()

    # Convert each user to a dictionary using the `as_dict` method
    users_list = [
        {key: value for key, value in user.as_dict().items() if key in [
            "id", "email", "first_name", "last_name", "mobile_number", "country", "created_at", "sweeps_points", "ban"]}
        for user in users
    ]

    return JSONResponse(content=users_list)


@app.get("/users/{user_id}")  # Define the route
def get_user_by_id(user_id: str, db: Session = Depends(get_db)):
    # Query the user by ID from the database
    user = db.query(User).filter(User.id == user_id).first()

    # Check if the user exists
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate total bets placed by the user
    bets_placed = db.query(Share).filter(Share.user_id == user_id).count()

    # Prepare the response data
    response_data = {
        "current_balance": user.sweeps_points,  # Assuming sweeps_points holds the current balance
        "bets_placed": bets_placed,  # Calculated from the Bet table
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": f"{user.first_name} {user.last_name}",  # Concatenate first and last name
        "email": user.email,
        "mobile_number": user.mobile_number,
        "address": user.address,
        "city": user.city,
        "state": user.state,
        "zip_postal": user.zip_postal,
        "country": user.country,
        "ban": user.ban,
    }

    return JSONResponse(content=response_data)


cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred)

@app.post("/register")
async def register(user: RegisterUser, db: Session = Depends(get_db)):
    try:
        # Create a user in Firebase
        user_record = auth.create_user(
            email=user.email,
            password=user.password
        )
        
        # Save user info in the local database
        db_user = User(
            id=user_record.uid,  # Using Firebase UID as the ID
            email=user.email,
            name=user.name or "",  # Provide default empty string if None
            first_name=user.first_name or "",  # Default empty string if None
            last_name=user.last_name or "",  # Default empty string if None
            mobile_number=user.mobile_number or "",  # Default empty string if None
            address=user.address or "",  # Default empty string if None
            city=user.city or "",  # Default empty string if None
            state=user.state or "",  # Default empty string if None
            zip_postal=user.zip_postal or "",  # Default empty string if None
            country=user.country or "",  # Default empty string if None
            role=user.role or "USER",  # Default role if None
            sweeps_points=1000.0,  # Default starting balance of Sweeps Points
            betting_points=1000.0,  # Default starting balance of Betting Points
            ban=False,  # Default value for ban status
        )
        
        db.add(db_user)
        db.commit()  # Commit the transaction to save the user
        db.refresh(db_user)  # Refresh to get the updated instance

        return {"message": "User registered successfully", "uid": user_record.uid}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

security = HTTPBearer()

@app.post("/login")
async def login(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify the Firebase ID Token
        decoded_token = auth.verify_id_token(token.credentials)
        uid = decoded_token.get("uid")

        # Firebase already checks expiration, so no need to manually check "exp"
        if not uid:
            raise HTTPException(status_code=400, detail="Token does not contain a UID")

        return {"message": "User authenticated", "uid": uid}

    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=403, detail="Token has expired")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Forbidden: {str(e)}")
    
# API to buy shares
@app.post("/api/market/buy-share")
async def buy_share(request: BuyShareRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.id == request.user_id).first()

    if request.outcome not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid outcome. Must be 1 or 0.")


    # Check the user's balance
    total_cost = (request.share_price/100) * request.shareCount
    if user.sweeps_points < total_cost:
        raise HTTPException(status_code=400, detail="Insufficient Sweeps Points balance.")

    # Deduct the cost from the user's balance
    user.sweeps_points -= total_cost

    # Determine the bet type based on the outcome
    bet_type = "yes" if request.outcome == 1 else "no"
    # Create a new share record
    new_share = Share(user_id=user.id, event_id=request.event_id, amount=request.shareCount, bet_type=bet_type)
    db.add(new_share)


    # Update the total yes or no bets based on the outcome
    if bet_type == "yes":
        event = db.query(Event).filter(Event.id == request.event_id).first()
        if event:
            event.total_yes_bets += request.shareCount
    else:
        event = db.query(Event).filter(Event.id == request.event_id).first()
        if event:
            event.total_no_bets += request.shareCount

    db.commit()

    return {"message": "Shares purchased successfully", "cost": total_cost}


# API to sell shares
@app.post("/api/market/sell-share")
async def sell_share(request: SellShareRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.user_id).first()

    if request.outcome not in [1, 0]:
        raise HTTPException(status_code=400, detail="Invalid outcome. Must be 1 or 0.")

    bet_type = "yes" if request.outcome == 1 else "no"

    # Retrieve shares held by the user
    shares = db.query(Share).filter(Share.user_id == user.id, Share.event_id == request.eventId, Share.bet_type == bet_type).all()
    total_shares = sum(share.amount for share in shares)

    if request.shareCount > total_shares:
        raise HTTPException(status_code=400, detail="Insufficient shares to sell.")


    # Credit the user's balance
    total_credits = request.share_price * request.shareCount
    user.sweeps_points += total_credits

    # Remove shares from the database
    # This is a simplistic way; you might want to adjust counts instead of deleting
    for share in shares:
        if share.amount >= request.shareCount:
            share.amount -= request.shareCount
            if share.amount == 0:
                db.delete(share)
            else:
                db.add(share)
    
    # Update the match's total bets
    if bet_type == 'yes':
        event = db.query(Event).filter(Event.id == request.eventId).first()
        if event:
            event.total_yes_bets -= request.shareCount
    else:
        event = db.query(Event).filter(Event.id == request.eventId).first()
        if event:
            event.total_no_bets -= request.shareCount

    db.commit()

    return {"message": "Shares sold successfully", "credits": total_credits}


@app.get("/api/market/share-price")
async def get_share_price(eventId: int, db: Session = Depends(get_db)):
    # Retrieve the match details for the given event ID
    match = db.query(Match).filter(Match.id == eventId).first()
    if not match:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Calculate the total shares bought for both outcomes
    total_shares = match.total_yes_bets + match.total_no_bets

    # Avoid division by zero if there are no shares bought yet
    if total_shares == 0:
        yes_price = 50  # Set a base price of 50 cents when no shares have been bought
        no_price = 50
    else:
        # Calculate share price as a percentage of total shares bought for each outcome
        yes_price = (match.total_yes_bets / total_shares) * 100
        no_price = (match.total_no_bets / total_shares) * 100

    return {
        "eventId": eventId,
        "yes_price": round(yes_price, 2),
        "no_price": round(no_price, 2)
    }


'''
@app.get("/api/market/user-shares")
async def get_user_shares(eventId: int,db: Session = Depends(get_db),token: HTTPAuthorizationCredentials = Depends(security)):
    # Decode token to get user ID
    decoded_token = auth.verify_id_token(token.credentials)
    user_id = decoded_token.get("uid")

    # Retrieve event details
    match = db.query(Match).filter(Match.id == eventId).first()
    if not match:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Calculate the total shares bought for both outcomes to determine current prices
    total_shares = match.total_yes_bets + match.total_no_bets

    if total_shares == 0:
        yes_price = 50  # Base price of 50 cents if no shares have been bought
        no_price = 50
    else:
        # Calculate share price in cents based on the proportion of total shares
        yes_price = (match.total_yes_bets / total_shares) * 100
        no_price = (match.total_no_bets / total_shares) * 100

    # Retrieve user's shares for the given event
    user_yes_shares = db.query(Share).filter(
        Share.user_id == user_id, Share.bet_id == eventId, Share.outcome == "Yes"
    ).first()
    user_no_shares = db.query(Share).filter(
        Share.user_id == user_id, Share.bet_id == eventId, Share.outcome == "No"
    ).first()

    # Calculate the quantity of shares owned by the user and their market value
    user_yes_quantity = user_yes_shares.amount if user_yes_shares else 0
    user_no_quantity = user_no_shares.amount if user_no_shares else 0

    user_yes_value = user_yes_quantity * (yes_price / 100)
    user_no_value = user_no_quantity * (no_price / 100)

    return {
        "eventId": eventId,
        "userShares": {
            "yes": {
                "quantity": user_yes_quantity,
                "market_value": round(user_yes_value, 2)
            },
            "no": {
                "quantity": user_no_quantity,
                "market_value": round(user_no_value, 2)
            }
        }
    }
'''


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


@app.patch("/api/admin/user-profile/edit/{userId}")
def edit_user_profile(userId: str, profile_data: UserProfileEdit, db: Session = Depends(get_db)):
    # Retrieve the user by ID
    user = db.query(User).filter(User.id == userId).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update profile fields if provided
    if profile_data.first_name:
        user.first_name = profile_data.first_name
    if profile_data.last_name:
        user.last_name = profile_data.last_name
    # If email is updated, update in Firebase Auth
    if profile_data.email and profile_data.email != user.email:
        try:
            # Update email in Firebase Authentication
            firebase_user = auth.get_user_by_email(user.email)
            auth.update_user(firebase_user.uid, email=profile_data.email)
        except auth.UserNotFoundError:
            raise HTTPException(status_code=404, detail="User not found in Firebase")
        except firebase_admin.exceptions.FirebaseError as e:
            raise HTTPException(status_code=400, detail=f"Error updating email in Firebase: {str(e)}")
    if profile_data.email:
        user.email = profile_data.email
    if profile_data.mobile_number:
        user.mobile_number = profile_data.mobile_number
    if profile_data.address:
        user.address = profile_data.address
    if profile_data.city:
        user.city = profile_data.city
    if profile_data.state:
        user.state = profile_data.state
    if profile_data.zip_postal:
        user.zip_postal = profile_data.zip_postal
    if profile_data.country:
        user.country = profile_data.country

    # Adjust balance
    if profile_data.add_balance:
        user.sweeps_points += profile_data.add_balance
    if profile_data.subtract_balance:
        if user.sweeps_points >= profile_data.subtract_balance:
            user.sweeps_points -= profile_data.subtract_balance
        else:
            raise HTTPException(status_code=400, detail="Insufficient balance for subtraction")

    # Update ban status if provided
    if profile_data.ban is not None:
        user.ban = profile_data.ban

    # Commit changes to the database
    db.commit()
    db.refresh(user)

   
    return {"message": "User profile updated successfully", "user_id": user.id}

# Dependency to verify the user using Firebase token
def get_current_user(token: str = Depends(HTTPBearer())):
    try:
        # Verify the Firebase token
        decoded_token = auth.verify_id_token(token.credentials)
        uid = decoded_token["uid"]
        return uid
    except Exception as e:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/api/user/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str, db: Session = Depends(get_db), current_user: str = Depends(get_current_user)):
    # Ensure the current user is authorized to access the requested profile (e.g., user can only access their own profile)
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Query the database for the user by their user_id
    user = db.query(User).filter(User.id == user_id).first()

    # If no user found, raise an exception
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Return the user profile as a response
    return user.as_dict()



@app.patch("/api/user/profile/edit/{user_id}", response_model=UserProfileEdit)
async def edit_user_profile(
    user_id: str, 
    profile_data: UserProfileEdit, 
    db: Session = Depends(get_db), 
    current_user: str = Depends(get_current_user)
):
    # Ensure the current user is authorized to edit their own profile
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Query the database for the user by their user_id
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update profile fields if provided
    if profile_data.first_name:
        user.first_name = profile_data.first_name
    if profile_data.last_name:
        user.last_name = profile_data.last_name
    if profile_data.email:
        user.email = profile_data.email
    if profile_data.mobile_number:
        user.mobile_number = profile_data.mobile_number
    if profile_data.address:
        user.address = profile_data.address
    if profile_data.city:
        user.city = profile_data.city
    if profile_data.state:
        user.state = profile_data.state
    if profile_data.zip_postal:
        user.zip_postal = profile_data.zip_postal
    if profile_data.country:
        user.country = profile_data.country

    # Commit changes to the database
    db.commit()
    db.refresh(user)

    return {"message": "Profile updated successfully", "user": user}


@app.patch("/api/user/profile/{user_id}/password")
async def change_password(
    user_id: str, 
    password_data: ChangePasswordRequest, 
    db: Session = Depends(get_db), 
    current_user: str = Depends(get_current_user)
):
    # Ensure the current user is authorized to change their own password
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Query the database for the user by their user_id
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update password in Firebase
    try:
        # Fetch user from Firebase by the user's ID
        firebase_user = auth.get_user(user.id)
        
        # Update the password in Firebase
        auth.update_user(firebase_user.uid, password=password_data.new_password)
    except auth.UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found in Firebase")
    except firebase_admin.exceptions.FirebaseError as e:
        raise HTTPException(status_code=400, detail=f"Error updating password in Firebase: {str(e)}")

    return {"message": "Password updated successfully"}



@app.post("/google_signup")
async def google_signup(authorization: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        # Get the Firebase token from the 'Authorization' header (Bearer token)
        id_token = authorization.credentials
        # Verify the Google ID token
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token.get("uid")

        # Get the user's info from Firebase
        user_record = auth.get_user(uid)

        # Get user's data (email, name, etc.)
        email = user_record.email
        full_name = user_record.display_name or ""  # Display name in Firebase
        first_name, last_name = "", ""

        if full_name:
            # Assuming full_name is in "First Last" format
            name_parts = full_name.split(" ", 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ""

        # Check if the user already exists in the local database by email or UID
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            return {"message": "User already exists", "uid": existing_user.id}

        # Create a new user in the local database if not exists
        db_user = User(
            id=uid,  # Using Firebase UID as the ID
            email=email,
            first_name=first_name,
            last_name=last_name,
            name=full_name,
            role="USER",  # Default role
            sweeps_points=1000.0,
            betting_points=1000.0,
            ban=False,
        )

        # Add user to the local database
        db.add(db_user)
        db.commit()  # Commit the transaction to save the user
        db.refresh(db_user)

        return {"message": "User signed up successfully", "uid": uid}

    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=400, detail="Invalid Google token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Schedule task to run every day at 12:10 PM
scheduler = BackgroundScheduler()

# Add a job to run scrape_and_store_matches at 12:10 PM every day
scheduler.add_job(scrape_and_store_matches, 'cron', hour=12, minute=10)

# Start the scheduler
scheduler.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
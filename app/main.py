# FastAPI Imports
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import uvicorn
# SQLAlchemy Imports for Database Models
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Firebase Admin Imports for Authentication
import firebase_admin
from firebase_admin import auth

# Pydantic Models
from typing import List

import json


# Custom Imports (e.g., utilities, helper functions)
from .db_config import (
    Base,
    get_db,
    engine,
)  # Import engine and SessionLocal from config.py
from .models import (
    User,
    Match,
    Event,
    Share,
    Remarks,
    RemarkType,
)  # Assuming these are your SQLAlchemy models
from .schemas import (
    MatchCreate,
    UserOut,
    EventResponse,
    CreateRemark,
    MatchBetUpdate,
    RegisterUser,
    BuyShareRequest,
    SellShareRequest,
    UserProfile,
    UserProfileEdit,
    ChangePasswordRequest,
)  # Assuming these are your Pydantic schemas
from .helper import scrape_and_store_matches

from .config import add_cors_middleware, start_scheduler
from .firebase import initialize_firebase

spread_value=2
# Create the tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
add_cors_middleware(app)

@app.on_event("startup")
async def startup_event():
    initialize_firebase()  # Initialize Firebase
    start_scheduler()      # Start scheduling tasks (like scraping)


@app.get("/api/scrape_and_store_matches")
def scrape_and_store_matches_route(db: Session = Depends(get_db)):
    return scrape_and_store_matches(db)


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
    db.refresh(
        db_match
    )  # Refresh to get the ID and other auto-generated fields from the database

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
        raise HTTPException(
            status_code=404, detail=f"Event with ID {event_id} not found"
        )

    # Query the associated match for this event
    db_match = db.query(Match).filter(Match.id == db_event.match_id).first()
    
    # If no match is found, raise an error
    if not db_match:
        raise HTTPException(
            status_code=404, detail=f"Match for event ID {event_id} not found"
        )
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
        "no": round(no_percentage, 2),
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
        "total_yes_bets": total_yes_bets+spread_value,
        "total_no_bets": total_no_bets+spread_value,
        "variations": db_event.variations,
        "match": {
            "id": db_match.id,
            "team1": db_match.team1,
            "team2": db_match.team2,
            "match_time": str(db_match.match_time),
            "league": db_match.league,
            "bet_start_time": str(db_match.bet_start_time),
            "bet_end_time": str(db_match.bet_end_time),
        },
    }

    return JSONResponse(content=response_data)


@app.get(
    "/events", response_model=List[EventResponse]
)  # Return a list of EventResponse objects
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
            event_data["match_time"] = (
                match.match_time
            )  # Include match_time from the Match table
            # Initialize variables to calculate percentages
            total_yes_bets = event.total_yes_bets
            total_no_bets = event.total_no_bets

            # Calculate yes/no percentages based on the current state
            if total_yes_bets + total_no_bets > 0:
                yes_percentage = (
                    total_yes_bets / (total_yes_bets + total_no_bets)
                ) * 100
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
            raise HTTPException(
                status_code=400, detail="Insufficient balance to subtract"
            )
        user.sweeps_points -= create_remark.amount

    # Add the new remark and update the user
    db.add(new_remark)
    db.commit()

    # Return response
    return {
        "message": "Remark added successfully",
        "user_id": create_remark.user_id,
        "amount": create_remark.amount,
    }


@app.post("/ban_unban")
def ban_unban_user(user_id: str, message: str, db: Session = Depends(get_db)):
    # Fetch the user from the database
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.ban = not (user.ban)

    # Create a new remark in the remarks table
    remark = Remarks(
        user_id=user.id,
        type=RemarkType.ban,  # Set the type as 'ban'
        message=message,
    )

    # Add the user ban and the remark to the session
    db.add(remark)
    db.commit()

    return {
        "message": f"User {user_id} has been banned successfully.",
        "user_id": user_id,
        "ban_status": user.ban,
    }


@app.post("/update-bets/")
def update_bets(match_list: List[MatchBetUpdate], db: Session = Depends(get_db)):
    for match in match_list:
        # Get the match by ID (use ORM model for querying)
        db_match = db.query(Match).filter(Match.id == match.id).first()

        if not db_match:
            raise HTTPException(
                status_code=404, detail=f"Match with ID {match.id} not found"
            )

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
            raise HTTPException(
                status_code=400, detail="Invalid bet value. Use 0 for no and 1 for yes."
            )

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
        {
            key: value
            for key, value in user.as_dict().items()
            if key
            in [
                "id",
                "email",
                "first_name",
                "last_name",
                "mobile_number",
                "country",
                "created_at",
                "sweeps_points",
                "ban",
            ]
        }
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


@app.post("/register")
async def register(user: RegisterUser, db: Session = Depends(get_db)):
    try:
        # Create a user in Firebase
        user_record = auth.create_user(email=user.email, password=user.password)

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


@app.post("/api/market/buy-share")
async def buy_share(request: BuyShareRequest, db: Session = Depends(get_db)):
    # Fetch user
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate bet_type and outcome
    if request.bet_type not in ["buy", "sell"]:
        raise HTTPException(
            status_code=400, detail="Invalid bet type. Must be 'buy' or 'sell'."
        )
    if request.outcome not in ["yes", "no"]:
        raise HTTPException(
            status_code=400, detail="Invalid outcome. Must be 'yes' or 'no'."
        )

    # Fetch event
    event = db.query(Event).filter(Event.id == request.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Fetch existing shares for the event
    existing_shares = (
        db.query(Share)
        .filter(Share.user_id == user.id, Share.event_id == request.event_id)
        .all()
    )

    # Check for conflicting outcomes
    for share in existing_shares:
        if share.outcome != request.outcome:
            raise HTTPException(
                status_code=400,
                detail=f"Conflicting outcome detected. Existing position is {share.outcome}.",
            )

    # Resolve opposing positions (e.g., sell existing buy positions)
    opposing_shares = [
        share
        for share in existing_shares
        if share.bet_type != request.bet_type and share.amount > 0
    ]
    remaining_shares = request.shareCount
    total_profit_or_loss = 0  # Track profit/loss for opposing trades

    for share in opposing_shares:
        trade_price = request.share_price  # Use the current share price for profit/loss calculation
        if share.amount >= remaining_shares:
            # Close partially or fully opposing position
            trade_amount = remaining_shares
            total_profit_or_loss += trade_amount * (request.share_price / 100 - share.share_price / 100)
            share.amount -= remaining_shares
            remaining_shares = 0
            db.add(share)
            break
        else:
            # Fully close the opposing position
            trade_amount = share.amount
            total_profit_or_loss += trade_amount * (request.share_price / 100 - share.share_price / 100)
            remaining_shares -= trade_amount
            db.delete(share)

    # Update user's balance based on profit/loss from opposing trades
    user.sweeps_points += total_profit_or_loss

    # Handle remaining shares (opening or updating position)
    if remaining_shares > 0:
        existing_position = next(
            (
                share
                for share in existing_shares
                if share.bet_type == request.bet_type and share.outcome == request.outcome
            ),
            None,
        )

        if existing_position:
            # Update the existing position
            existing_position.amount += remaining_shares
            # Update the limit price if provided
            if request.limit_price:
                existing_position.limit_price = request.limit_price
            db.add(existing_position)
        else:
            # Create a new position
            new_share = Share(
                user_id=user.id,
                event_id=request.event_id,
                amount=remaining_shares,
                bet_type=request.bet_type,
                outcome=request.outcome,
                share_price=request.share_price,  # Store the current share price
                limit_price=request.limit_price,  # Save limit price if provided
            )
            db.add(new_share)

        # Deduct cost from user's balance for new buy positions
        if request.bet_type == "buy":
            total_cost = (request.share_price / 100) * remaining_shares
            if user.sweeps_points < total_cost:
                raise HTTPException(
                    status_code=400, detail="Insufficient balance for the trade."
                )
            user.sweeps_points -= total_cost

    # Update event totals
    if request.outcome == "yes":
        if request.bet_type == "buy":
            event.total_yes_bets += request.shareCount
        else:
            event.total_yes_bets -= request.shareCount
    elif request.outcome == "no":
        if request.bet_type == "buy":
            event.total_no_bets += request.shareCount
        else:
            event.total_no_bets -= request.shareCount

    # Commit transaction
    db.commit()
    return {
        "message": "Trade executed successfully",
        "profit_or_loss": round(total_profit_or_loss, 2),
    }


@app.post("/api/market/sell-share")
async def sell_share(request: SellShareRequest, db: Session = Depends(get_db)):
    # Fetch user
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate bet_type and outcome
    if request.bet_type != "sell":
        raise HTTPException(
            status_code=400, detail="Invalid bet type for selling. Must be 'sell'."
        )
    if request.outcome not in ["yes", "no"]:
        raise HTTPException(
            status_code=400, detail="Invalid outcome. Must be 'yes' or 'no'."
        )

    # Fetch event
    event = db.query(Event).filter(Event.id == request.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Fetch existing shares for the event
    existing_shares = (
        db.query(Share)
        .filter(Share.user_id == user.id, Share.event_id == request.event_id)
        .all()
    )

    # Check for conflicting outcomes
    for share in existing_shares:
        if share.outcome != request.outcome:
            raise HTTPException(
                status_code=400,
                detail=f"Conflicting outcome detected. Existing position is {share.outcome}.",
            )

    # Resolve opposing positions (e.g., buy existing sell positions)
    opposing_shares = [
        share
        for share in existing_shares
        if share.bet_type == "buy" and share.amount > 0
    ]
    remaining_shares = request.shareCount
    total_profit_or_loss = 0  # Track profit/loss for opposing trades

    for share in opposing_shares:
        trade_price = request.share_price  # Use the current share price for profit/loss calculation
        if share.amount >= remaining_shares:
            # Close partially or fully opposing position
            trade_amount = remaining_shares
            total_profit_or_loss += trade_amount * (share.share_price / 100 - request.share_price / 100)
            share.amount -= remaining_shares
            remaining_shares = 0
            db.add(share)
            break
        else:
            # Fully close the opposing position
            trade_amount = share.amount
            total_profit_or_loss += trade_amount * (share.share_price / 100 - request.share_price / 100)
            remaining_shares -= trade_amount
            db.delete(share)

    # Update user's balance based on profit/loss from opposing trades
    user.sweeps_points += total_profit_or_loss

    # Handle remaining shares (opening or updating position)
    if remaining_shares > 0:
        existing_position = next(
            (
                share
                for share in existing_shares
                if share.bet_type == "sell" and share.outcome == request.outcome
            ),
            None,
        )

        if existing_position:
            # Update the existing position
            existing_position.amount += remaining_shares
            # Update the limit price if provided
            if request.limit_price:
                existing_position.limit_price = request.limit_price
            db.add(existing_position)
        else:
            # Create a new position
            new_share = Share(
                user_id=user.id,
                event_id=request.event_id,
                amount=remaining_shares,
                bet_type="sell",
                outcome=request.outcome,
                share_price=request.share_price,  # Store the current share price
                limit_price=request.limit_price,  # Save limit price if provided
            )
            db.add(new_share)

    # Update event totals
    if request.outcome == "yes":
        event.total_yes_bets -= request.shareCount
    elif request.outcome == "no":
        event.total_no_bets -= request.shareCount

    # Commit transaction
    db.commit()
    return {
        "message": "Trade executed successfully",
        "profit_or_loss": round(total_profit_or_loss, 2),
    }


@app.get("/api/market/share-price")
async def get_share_price(eventId: int, type: str, db: Session = Depends(get_db)):
    # Validate the type parameter
    if type not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Invalid type. Must be 'buy' or 'sell'.")

    # Retrieve the match details for the given event ID
    match = db.query(Event).filter(Event.id == eventId).first()
    if not match:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Calculate the total shares bought for both outcomes
    total_shares = match.total_yes_bets + match.total_no_bets

    # Set a fixed spread value
    spread_value = 2  # Fixed spread value

    # Avoid division by zero if there are no shares bought yet
    if total_shares == 0:
        yes_price = 50  # Set a base price of 50 when no shares have been bought
        no_price = 50
    else:
        # Calculate share price as a percentage of total shares bought for each outcome
        yes_price = (match.total_yes_bets / total_shares) * 100
        no_price = (match.total_no_bets / total_shares) * 100

    # Adjust prices based on the type (buy or sell)
    if type == "buy":
        yes_price += spread_value
        no_price += spread_value
    elif type == "sell":
        yes_price -= spread_value
        no_price -= spread_value

    # Ensure prices don't go below 0
    yes_price = max(0, yes_price)
    no_price = max(0, no_price)

    return {
        "eventId": eventId,
        "type": type,
        "yes_price": round(yes_price, 2),
        "no_price": round(no_price, 2),
    }


"""
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
"""


@app.patch("/api/admin/user-profile/edit/{userId}")
def edit_user_profile(
    userId: str, profile_data: UserProfileEdit, db: Session = Depends(get_db)
):
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
            raise HTTPException(
                status_code=400, detail=f"Error updating email in Firebase: {str(e)}"
            )
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
            raise HTTPException(
                status_code=400, detail="Insufficient balance for subtraction"
            )

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
async def get_user_profile(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
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
    current_user: str = Depends(get_current_user),
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
    current_user: str = Depends(get_current_user),
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
        raise HTTPException(
            status_code=400, detail=f"Error updating password in Firebase: {str(e)}"
        )

    return {"message": "Password updated successfully"}


@app.post("/google_signup")
async def google_signup(
    authorization: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

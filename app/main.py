# FastAPI Imports
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import uvicorn

# SQLAlchemy Imports for Database Models
from sqlalchemy import func
from sqlalchemy.orm import Session

# Firebase Admin Imports for Authentication
import firebase_admin
from firebase_admin import auth

from datetime import datetime

# Pydantic Models
from typing import List, Dict

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
    UserOut,
    EventResponse,
    CreateRemark,
    RegisterUser,
    BuyShareRequest,
    SellShareRequest,
    UserProfile,
    UserProfileEdit,
    ChangePasswordRequest,
    EventDetailResponse,
    MatchResponse
)  # Assuming these are your Pydantic schemas
from .helper import fetch_and_store_matches
from .config import add_cors_middleware, start_scheduler
from .firebase import initialize_firebase
from collections import defaultdict
from sqlalchemy.orm.attributes import flag_modified


spread_value = 2
# Create the tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
add_cors_middleware(app)


@app.on_event("startup")
async def startup_event():
    initialize_firebase()  # Initialize Firebase
    start_scheduler()  # Start scheduling tasks (like scraping)


@app.get("/api/fetch_and_store_matches")
def fetch_and_store_matches_route(db: Session = Depends(get_db)):
    return fetch_and_store_matches(db)


# API to retrieve all matches
@app.get("/matches", response_model=Dict[str, List[MatchResponse]])
def get_matches(db: Session = Depends(get_db)):
    matches = db.query(Match).all()
    
    grouped_matches = {}
    for match in matches:
        sport = match.sport
        match_data = MatchResponse.from_orm(match)
        
        if sport not in grouped_matches:
            grouped_matches[sport] = []
            
        
        grouped_matches[sport].append(match_data.dict())

    return grouped_matches


@app.get("/matches/basketball", response_model=Dict[str, List[MatchResponse]])
def get_basketball_matches(db: Session = Depends(get_db)):
    """
    Get all upcoming basketball matches grouped by league.
    """
    current_time = datetime.utcnow()

    # Fetch all basketball matches with match_time in the future
    basketball_matches = (
        db.query(Match)
        .filter(Match.sport == "basketball", Match.bet_end_time > current_time)
        .order_by(Match.league, Match.match_time)
        .all()
    )

    if not basketball_matches:
        raise HTTPException(
            status_code=404,
            detail="No upcoming basketball matches found"
        )

    # Group matches by league
    matches_by_league = {}
    for match in basketball_matches:
        league = match.league
        match_data = MatchResponse(
            id=match.id,
            sport=match.sport,
            league=match.league,
            team1=match.team1,
            team2=match.team2,
            match_time=match.match_time.isoformat(),
            bet_end_time=match.bet_end_time
        )
        if league not in matches_by_league:
            matches_by_league[league] = []
        matches_by_league[league].append(match_data)

    return matches_by_league


@app.get("/event/{event_id}")
def get_event_by_id(event_id: int, db: Session = Depends(get_db)):

    """
    Get detailed information for a specific event by its ID.
    """
    # Fetch the event and associated match
    db_event = db.query(Event).filter(Event.id == event_id).first()
    if not db_event:
        raise HTTPException(status_code=404, detail=f"Event with ID {event_id} not found")

    db_match = db.query(Match).filter(Match.id == db_event.match_id).first()
    if not db_match:
        raise HTTPException(status_code=404, detail=f"Match for event ID {event_id} not found")

    # Ensure variations' timestamp is converted to datetime before formatting
    variations_with_iso_timestamps = []
    for variation in db_event.variations:
        try:
            # Convert timestamp to datetime if it's a string
            timestamp = variation["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)  # Convert to datetime
            
            variations_with_iso_timestamps.append({
                "timestamp": timestamp.isoformat(),
                "buy_price": variation["buy_price"],
                "sell_price": variation["sell_price"]
            })
        except Exception as e:
            print(f"Error parsing timestamp: {e}")  # Debugging purpose, remove in production

    # Build response (NO MODIFICATIONS!)
    response_data = {
        "id": db_event.id,
        "match_id": db_event.match_id,
        "question": db_event.question,
        "type": db_event.type,
        "threshold": db_event.threshold,
        "buy_sell_index": db_event.buy_sell_index,
        "buy_price": round(db_event.buy_price, 2),
        "sell_price": round(db_event.sell_price, 2),
        "variations": variations_with_iso_timestamps,  # Graph tracking
        "match": {
            "id": db_match.id,
            "team1": db_match.team1,
            "team2": db_match.team2,
            "match_time": db_match.match_time.isoformat(),
            "league": db_match.league,
        },
    }

    return JSONResponse(content=response_data)


@app.get("/events/{match_id}", response_model=Dict[str, List[EventResponse]])
def get_events_by_match_id(match_id: int, db: Session = Depends(get_db)):
    """
    Get all events for a specific match ID, grouped by heading.
    """
    # Fetch events associated with the given match ID
    events = db.query(Event).filter(Event.match_id == match_id).all()

    if not events:
        raise HTTPException(
            status_code=404,
            detail=f"No events found for match ID {match_id}"
        )

    grouped_events = defaultdict(list)

    for event in events:
        match = event.match
        if not match:
            continue  # Skip events with no associated match

        # Construct the EventResponse object
        event_data = EventResponse(
            id=event.id,
            match_id=event.match_id,
            question=event.question,
            type=event.type or "N/A",
            heading=event.heading,
            threshold=event.threshold or 0.0,
            buy_sell_index=event.buy_sell_index,
            buy_price=round(event.buy_price, 2),
            sell_price=round(event.sell_price, 2),
            variations=event.variations,
            match_time=match.match_time.isoformat(),
            sport=match.sport,
            league=match.league,
            team1=match.team1,
            team2=match.team2,
        )

        # Group by heading
        grouped_events[event.heading].append(event_data)

    return grouped_events


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
    """
    Execute a buy trade for an event at any price, offsetting any existing sell positions first.
    """
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    event = db.query(Event).filter(Event.id == request.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    existing_sell_shares = db.query(Share).filter(
        Share.user_id == user.id, 
        Share.event_id == request.event_id, 
        Share.bet_type == "sell"
    ).all()

    remaining_shares = request.shareCount
    total_profit_or_loss = 0  
    variations = []  

    # 🛠 **Offset Sell Positions First**
    for share in existing_sell_shares:
        if share.amount >= remaining_shares:
            total_profit_or_loss += remaining_shares * (share.share_price / 100 - request.share_price / 100)
            share.amount -= remaining_shares
            remaining_shares = 0
            db.add(share)
            break
        else:
            total_profit_or_loss += share.amount * (share.share_price / 100 - request.share_price / 100)
            remaining_shares -= share.amount
            db.delete(share)

    if total_profit_or_loss != 0:
        variations.append({
            "timestamp": datetime.now().isoformat(),  
            "buy_price": round(event.buy_price, 2),
            "sell_price": round(event.sell_price, 2),
            "note": "Offset trade executed"
        })

    user.sweeps_points += total_profit_or_loss

    if remaining_shares > 0:
        total_cost = (request.share_price / 100) * remaining_shares
        if user.sweeps_points < total_cost:
            raise HTTPException(status_code=400, detail="Insufficient balance for the trade.")
        user.sweeps_points -= total_cost

        new_share = Share(
            user_id=user.id,
            event_id=request.event_id,
            amount=remaining_shares,
            bet_type="buy",
            share_price=request.share_price,
            limit_price=request.limit_price,
        )
        db.add(new_share)

        spread_factor = 2
        max_buy_price = 100
        event.buy_price = min(event.buy_price + (remaining_shares / 10), max_buy_price)
        event.sell_price = max(event.buy_price - spread_factor, 51)

        variations.append({
            "timestamp": datetime.now().isoformat(),  
            "buy_price": round(event.buy_price, 2),
            "sell_price": round(event.sell_price, 2),
            "note": "New buy order executed"
        })

    event.variations.extend(variations)  
    flag_modified(event, "variations")
    db.add(event)
    db.commit()
    
    return {
        "message": "Buy trade executed successfully",
        "profit_or_loss": round(total_profit_or_loss, 2),
        "buy_price": round(event.buy_price, 2),
        "sell_price": round(event.sell_price, 2),
    }


@app.post("/api/market/sell-share")
async def sell_share(request: SellShareRequest, db: Session = Depends(get_db)):
    """
    Execute a sell trade for an event at any price, offsetting any existing buy positions first.
    """
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    event = db.query(Event).filter(Event.id == request.event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    existing_buy_shares = db.query(Share).filter(
        Share.user_id == user.id, 
        Share.event_id == request.event_id, 
        Share.bet_type == "buy"
    ).all()

    remaining_shares = request.shareCount
    total_profit_or_loss = 0  
    variations = []  

    # 🛠 **Offset Buy Positions First**
    for share in existing_buy_shares:
        if share.amount >= remaining_shares:
            total_profit_or_loss += remaining_shares * (request.share_price / 100 - share.share_price / 100)
            share.amount -= remaining_shares
            remaining_shares = 0
            db.add(share)
            break
        else:
            total_profit_or_loss += share.amount * (request.share_price / 100 - share.share_price / 100)
            remaining_shares -= share.amount
            db.delete(share)

    if total_profit_or_loss != 0:
        variations.append({
            "timestamp": datetime.now().isoformat(),  
            "buy_price": round(event.buy_price, 2),
            "sell_price": round(event.sell_price, 2),
            "note": "Offset trade executed"
        })

    user.sweeps_points += total_profit_or_loss

    if remaining_shares > 0:
        total_cost = (request.share_price / 100) * remaining_shares
        if user.sweeps_points < total_cost:
            raise HTTPException(status_code=400, detail="Insufficient balance for the trade.")
        user.sweeps_points -= total_cost

        new_share = Share(
            user_id=user.id,
            event_id=request.event_id,
            amount=remaining_shares,
            bet_type="sell",
            share_price=request.share_price,
            limit_price=request.limit_price,
        )
        db.add(new_share)

        spread_factor = 2
        event.buy_price = max(event.buy_price - (remaining_shares / 10), 51)
        event.sell_price = event.buy_price - spread_factor

        variations.append({
            "timestamp": datetime.now().isoformat(),
            "buy_price": round(event.buy_price, 2),
            "sell_price": round(event.sell_price, 2),
            "note": "New sell order executed"
        })

    event.variations.extend(variations)  
    flag_modified(event, "variations")

    db.add(event)
    db.commit()

    return {
        "message": "Sell trade executed successfully",
        "profit_or_loss": round(total_profit_or_loss, 2),
        "buy_price": round(event.buy_price, 2),
        "sell_price": round(event.sell_price, 2),
    }


#useless prolly
@app.get("/api/market/share-price")
async def get_share_price(eventId: int, type: str, db: Session = Depends(get_db)):
    # Validate the type parameter
    if type not in ["buy", "sell"]:
        raise HTTPException(
            status_code=400, detail="Invalid type. Must be 'buy' or 'sell'."
        )

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


@app.get("/api/ai-bets", response_model=list)
def get_ai_bets(db: Session = Depends(get_db)):
    """
    API to fetch bets placed by the AI bot.
    Returns a list of bets with team names, league, bet type, outcome, number of shares, and bet time.
    """

    bot_user_id = "8AAjT02uf3XwQuHaebR4bTacbw92"  # AI bot user ID
    
    # Query to fetch all bets made by the AI bot
    ai_bets = (
        db.query(Share)
        .join(Event, Share.event_id == Event.id)
        .join(Match, Event.match_id == Match.id)
        .filter(Share.user_id == bot_user_id)
        .all()
    )

    if not ai_bets:
        raise HTTPException(status_code=404, detail="No bets found for the AI bot.")

    # Prepare response
    response = []
    for bet in ai_bets:
        response.append({
            "team1": bet.event.match.team1,
            "team2": bet.event.match.team2,
            "league": bet.event.match.league,
            "bet_type": bet.bet_type,
            "number_of_shares": bet.amount,
            "bet_time": bet.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })

    return response


@app.get("/events/results", response_model=List[EventDetailResponse])
def get_event_details(db: Session = Depends(get_db)):
    # Query all events and their associated matches
    events = (
        db.query(Event)
        .join(Match, Match.id == Event.match_id)
        .outerjoin(Share, Share.event_id == Event.id)
        .all()
    )
    
    if not events:
        raise HTTPException(status_code=404, detail="No events found.")

    event_details = []
    for event in events:
        total_bets = db.query(Share).filter(Share.event_id == event.id).count()
        match = db.query(Match).filter(Match.id == event.match_id).first()
        
        if not match:
            continue  # Skip if match information is missing

        event_details.append({
            "question": event.question,
            "team1": match.team1,
            "team2": match.team2,
            "bet_end_time": str(match.bet_end_time),
            "total_bets": total_bets,
            "resolved": event.resolved,
            "winner": event.winner,
        })

    return event_details


@app.post("/api/add_dummy_ai_user", response_model=dict)
def add_dummy_ai_user(db: Session = Depends(get_db)):
    """
    API to add a dummy AI user for betting purposes.
    If the user already exists, return the existing user.
    """
    bot_user_id = "8AAjT02uf3XwQuHaebR4bTacbw92"  # Static ID for AI bot user

    # Check if AI user already exists
    existing_user = db.query(User).filter(User.id == bot_user_id).first()
    if existing_user:
        return {"message": "AI user already exists", "user_id": bot_user_id}

    # Create new AI user
    ai_user = User(
        id=bot_user_id,
        name="AI_Betting_Bot",
        email="ai_betting@dummy.com",
        sweeps_points=100000.0,  # Give some initial balance for betting
    )

    db.add(ai_user)
    db.commit()
    db.refresh(ai_user)

    return {"message": "AI user created successfully", "user_id": ai_user.id}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


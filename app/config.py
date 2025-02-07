from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi.middleware.cors import CORSMiddleware
from .helper import fetch_and_store_matches
from .db_config import get_db
from .helper import calculate_results_for_event  # Ensure this is imported correctly
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from .models import Match, Event, Share
from .helper import calculate_share_price, ai_place_bet


logging.basicConfig(level=logging.INFO)


# FastAPI app configuration
def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def run_fetch_and_store_matches():
    # Get a session from the DB
    db = next(get_db())
    fetch_and_store_matches(db)


def check_and_process_results(db: Session):
    """
    Process trade results for finished matches and all their associated events.
    - Finds all events linked to matches that are marked as "finished".
    - Ensures results are calculated only once per event.
    """
    finished_events = (
        db.query(Event)
        .join(Match)
        .filter(Match.status == "finished", Event.resolved == False)
        .all()
    )

    for event in finished_events:
        logging.info(f"Processing event ID: {event.id}")
        try:
            calculate_results_for_event(event.id, db)
            logging.info(f"Results processed for event ID: {event.id}")
        except ValueError as e:
            logging.warning(f"Event ID {event.id}: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing event ID {event.id}: {str(e)}")

    return {"message": "All eligible events processed"}


def process_results_job():
    """Wrapper function to create a new DB session and pass it to check_and_process_results."""
    db = next(get_db())
    try:
        check_and_process_results(db)  # ✅ Pass 'db' argument
    finally:
        db.close()  # ✅ Close session to prevent memory leaks


def execute_stop_orders():
    """Check for limit conditions and execute stop orders."""
    db: Session = next(get_db())

    # Fetch all pending shares with a limit price
    pending_shares = db.query(Share).filter(Share.limit_price != None).all()

    for share in pending_shares:
        # Retrieve the current market price for this event
        market_prices = calculate_share_price(share.event_id, db)
        market_price = market_prices["buy_price"] if share.bet_type == "buy" else market_prices["sell_price"]

        # Check limit conditions
        if share.bet_type == "buy" and market_price <= share.limit_price:
            # Execute the buy trade
            execute_trade(share, market_price, db)

        elif share.bet_type == "sell" and market_price >= share.limit_price:
            # Execute the sell trade
            execute_trade(share, market_price, db)

    db.commit()


def execute_trade(share, market_price, db):
    """Execute the trade based on the share and market price."""
    user = share.user

    # Process Buy Trades
    if share.bet_type == "buy":
        total_cost = market_price * share.amount / 100
        if user.sweeps_points < total_cost:
            print(f"User {user.id} has insufficient balance to execute buy trade.")
            return
        user.sweeps_points -= total_cost

    # Process Sell Trades
    elif share.bet_type == "sell":
        total_revenue = market_price * share.amount / 100
        user.sweeps_points += total_revenue

    # Remove the share from the database after executing the trade
    db.delete(share)
    db.commit()


def run_ai_betting():
    """
    Periodically checks for eligible events and places bets as the AI bot.
    """
    db = next(get_db())
    try:
        # Get current time
        current_time = datetime.utcnow()

        # Find eligible events where betting is still open
        eligible_events = (
            db.query(Event)
            .join(Match)
            .filter(Match.match_time > current_time)  # Betting still open
            .all()
        )

        # Place bets for each eligible event
        for event in eligible_events:
            ai_place_bet(event.id, db)

    except Exception as e:
        logging.error(f"Error in AI betting: {str(e)}")
    finally:
        db.close()


def start_scheduler():
    scheduler = BackgroundScheduler()

    # Scraping job
    scheduler.add_job(
        lambda: run_fetch_and_store_matches(), "cron", hour=12, minute=10
    )

    # Result calculation job
    scheduler.add_job(
        process_results_job,
        IntervalTrigger(minutes=2),
        id="result_scheduler",
        replace_existing=True,
    )

    scheduler.add_job(lambda: execute_stop_orders(), "interval", minutes=10)

    scheduler.add_job(
    run_ai_betting,  # The function to execute
    "interval", 
    minutes=30, 
    id="ai_betting_scheduler", 
    replace_existing=True
    )


    scheduler.start()
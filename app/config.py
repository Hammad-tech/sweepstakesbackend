from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi.middleware.cors import CORSMiddleware
from .helper import scrape_and_store_matches
from .db_config import get_db
from .helper import calculate_results_for_event  # Ensure this is imported correctly
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from .models import Match, Event, Share
from .helper import calculate_share_price


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


def run_scrape_and_store_matches():
    # Get a session from the DB
    db = next(get_db())
    scrape_and_store_matches(db)


def check_and_process_results():
    # Start a new database session
    db: Session = next(get_db())
    try:
        # Fetch events eligible for result calculation
        eligible_events = (
            db.query(Event)
            .join(Match)
            .filter(Match.match_time <= datetime.utcnow() - timedelta(hours=3))
            .all()
        )

        for event in eligible_events:
            logging.info(f"Processing event ID: {event.id}")
            try:
                calculate_results_for_event(event.id, db)
                logging.info(f"Results calculated for event ID: {event.id}")
            except ValueError as e:
                logging.warning(f"Event ID {event.id}: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing event ID {event.id}: {str(e)}")

    finally:
        db.close()


def execute_stop_orders():
    """Check for limit conditions and execute stop orders."""
    db: Session = next(get_db())

    # Fetch all pending shares with a limit price
    pending_shares = db.query(Share).filter(Share.limit_price != None).all()

    for share in pending_shares:
        # Determine the type (buy or sell) for share price retrieval
        bet_type = "buy" if share.bet_type == "buy" else "sell"

        # Retrieve the current share prices
        market_prices = calculate_share_price(share.event_id, bet_type, db)
        yes_price = market_prices["yes_price"]
        no_price = market_prices["no_price"]

        # Determine the market price to check against
        if share.outcome == "yes":
            market_price = yes_price
        elif share.outcome == "no":
            market_price = no_price
        else:
            continue  # Skip invalid outcomes

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

    if share.bet_type == "buy":
        # Deduct the total cost from the user's balance
        total_cost = market_price * share.amount / 100
        if user.sweeps_points < total_cost:
            print(f"User {user.id} has insufficient balance to execute buy trade.")
            return
        user.sweeps_points -= total_cost
    elif share.bet_type == "sell":
        # Add the total revenue to the user's balance
        total_revenue = market_price * share.amount / 100
        user.sweeps_points += total_revenue

    # Remove the share from the database after executing the trade
    db.delete(share)
    db.commit()


def start_scheduler():
    scheduler = BackgroundScheduler()

    # Scraping job
    scheduler.add_job(
        lambda: run_scrape_and_store_matches(), "cron", hour=12, minute=10
    )

    # Result calculation job
    scheduler.add_job(
        check_and_process_results,
        IntervalTrigger(minutes=30),
        id="result_scheduler",
        replace_existing=True,
    )

    scheduler.add_job(lambda: execute_stop_orders(), "interval", minutes=10)

    scheduler.start()

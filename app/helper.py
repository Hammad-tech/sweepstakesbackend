from sqlalchemy.orm import Session  # Import Session
from datetime import datetime, timedelta

import time as t
from typing import List
from .models import Match, Event, Share, User, EventType
from .schemas import BuyShareRequest
import os
import requests
from sqlalchemy import func
from dotenv import load_dotenv
from fastapi import HTTPException
import joblib
import pandas as pd
from random import randint
import logging
from sqlalchemy.orm.attributes import flag_modified


def calculate_share_price(event_id: int, db: Session):
    """
    Calculate the share price for a given event based on buy/sell trade volume.

    Args:
        event_id (int): The ID of the event.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the buy and sell prices.
    """
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Retrieve the latest buy and sell price from event variations
    latest_variation = event.variations[-1] if event.variations else None

    if latest_variation:
        buy_price = latest_variation["buy_price"]
        sell_price = latest_variation["sell_price"]
    else:
        buy_price = 50  # Default price if no previous variations exist
        sell_price = 50

    return {
        "event_id": event_id,
        "buy_price": round(buy_price, 2),
        "sell_price": round(sell_price, 2),
    }


def calculate_results_for_event(event_id: int, db: Session):
    """
    Process buy/sell trades for a finished event.
    - Determines the winner from stored match results.
    - Settles all shares based on bet type and actual outcome.
    """
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise ValueError(f"Event with ID {event_id} not found.")

    match = db.query(Match).filter(Match.id == event.match_id).first()
    if not match or match.status != "finished":
        raise ValueError(f"Match associated with event ID {event_id} is not finished.")

    # ✅ Retrieve stored match result
    match_result = match.result  # Stored JSON with scores
    if not match_result:
        raise ValueError("Match result not available in database.")

    home_team = match.team1
    away_team = match.team2
    home_score = match_result["scores"]["home"]["total"]
    away_score = match_result["scores"]["away"]["total"]

    # ✅ Determine winner for match result events
    winning_team = home_team if home_score > away_score else away_team if away_score > home_score else "draw"

    # ✅ Fetch all shares (bets) for this event
    shares = db.query(Share).filter(Share.event_id == event_id).all()

    for share in shares:
        user = db.query(User).filter(User.id == share.user_id).first()
        if not user:
            continue

        # ✅ Process different event types
        if event.type == EventType.MATCH_RESULT:
            bet_team = event.question.split(" to Win?")[0]

            # ✅ Process Buy/Sell for Match Result
            if bet_team == winning_team:
                # Buy bet wins, Sell bet loses
                if share.bet_type == "buy":
                    profit = share.amount * (100 - share.share_price) / 100
                    user.sweeps_points += profit
                elif share.bet_type == "sell":
                    loss = share.amount * (share.share_price) / 100
                    user.sweeps_points -= loss
            else:
                # Buy bet loses, Sell bet wins
                if share.bet_type == "buy":
                    loss = share.amount * (share.share_price) / 100
                    user.sweeps_points -= loss
                elif share.bet_type == "sell":
                    profit = share.amount * (100 - share.share_price) / 100
                    user.sweeps_points += profit

        elif event.type == EventType.OVER_UNDER:
            # ✅ Process Buy/Sell for Over/Under Events
            if "Total Points" in event.heading:
                total_points = home_score + away_score
            else:
                team_name = event.heading.split(" Points")[0]
                total_points = home_score if team_name == home_team else away_score

            bet_wins = total_points > event.threshold
            if bet_wins:
                # Buy bet wins, Sell bet loses
                if share.bet_type == "buy":
                    profit = share.amount * (100 - share.share_price) / 100
                    user.sweeps_points += profit
                elif share.bet_type == "sell":
                    loss = share.amount * (share.share_price) / 100
                    user.sweeps_points -= loss
            else:
                # Buy bet loses, Sell bet wins
                if share.bet_type == "buy":
                    loss = share.amount * (share.share_price) / 100
                    user.sweeps_points -= loss
                elif share.bet_type == "sell":
                    profit = share.amount * (100 - share.share_price) / 100
                    user.sweeps_points += profit

        elif event.type == EventType.HANDICAP:
            # ✅ Process Buy/Sell for Handicap Events
            handicap_value = event.threshold
            adjusted_home_score = home_score + handicap_value
            bet_wins = adjusted_home_score > away_score

            if bet_wins:
                # Buy bet wins, Sell bet loses
                if share.bet_type == "buy":
                    profit = share.amount * (100 - share.share_price) / 100
                    user.sweeps_points += profit
                elif share.bet_type == "sell":
                    loss = share.amount * (share.share_price) / 100
                    user.sweeps_points -= loss
            else:
                # Buy bet loses, Sell bet wins
                if share.bet_type == "buy":
                    loss = share.amount * (share.share_price) / 100
                    user.sweeps_points -= loss
                elif share.bet_type == "sell":
                    profit = share.amount * (100 - share.share_price) / 100
                    user.sweeps_points += profit

        # ✅ Remove resolved trades
        db.delete(share)

    # ✅ Mark event as resolved
    event.resolved = True
    event.winner = winning_team
    db.add(event)

    db.commit()
    return {"message": f"Results processed successfully for event {event_id}"}


def buy_share(request: BuyShareRequest, db):
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


def get_share_price(event_id: int, db):
    """
    Fetches the latest buy/sell prices for an event.

    Args:
        event_id (int): The ID of the event.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing buy and sell prices.
    """
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Retrieve latest buy and sell prices from event variations
    latest_variation = event.variations[-1] if event.variations else None

    if latest_variation:
        buy_price = latest_variation["buy_price"]
        sell_price = latest_variation["sell_price"]
    else:
        buy_price = 50  # Default if no variations exist
        sell_price = 50

    return {
        "event_id": event_id,
        "buy_price": round(buy_price, 2),
        "sell_price": round(sell_price, 2),
    }


model = joblib.load("app/team_matchup_predictor.pkl")
scaler = joblib.load("app/scaler.pkl")
stats_df = pd.read_excel('app/team_stats_data.xlsx')


def get_team_stats(team_name: str):
    """
    Extract stats for a specific team based on required features.
    """

    features = [
    "ADJ OE", "ADJ DE", "EFG", "EFG D", "FT RATE", "FT RATE D", 
    "TOV%", "TOV% D", "O REB%", "OP OREB%", "2P %", "2P % D.", "3P %", "3P % D."
    ]
    try:
        # Filter the dataset for the specific team
        team_stats = stats_df.loc[stats_df['TEAM'] == team_name, features]
        
        if team_stats.empty:
            raise ValueError(f"Team {team_name} not found in the dataset.")
        
        # Return the team stats as a dictionary
        return team_stats.iloc[0].values  # Convert the row to a dictionary
    except Exception as e:
        raise ValueError(f"Error fetching stats for {team_name}: {str(e)}")


def predict_team_win_probability(team1, team2):
    """
    Predict the win probabilities for two teams.
    """

    team1_stats= get_team_stats(team1)
    team2_stats= get_team_stats(team2)

    # Scale the stats
    team1_scaled = scaler.transform([team1_stats])
    team2_scaled = scaler.transform([team2_stats])

    # Predict win rates
    team1_win_rate = model.predict(team1_scaled)[0]
    team2_win_rate = model.predict(team2_scaled)[0]

    # Normalize probabilities
    total = team1_win_rate + team2_win_rate
    team1_prob = team1_win_rate / total
    team2_prob = team2_win_rate / total

    return team1_prob, team2_prob


def predict(team1: str, team2: str):
    try:
        # Check if teams exist in the dataset
        if team1 not in stats_df['TEAM'].values or team2 not in stats_df['TEAM'].values:
            raise ValueError("One or both teams not found in the dataset.")
        
        # Predict the match-up
        team1_prob,team2_prob = predict_team_win_probability(team1, team2)
        return team1_prob,team2_prob
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def ai_place_bet(event_id: int, db):
    """
    AI Bot places bets for a given event intelligently.
    - Predicts probabilities for the two teams using the trained model.
    - Bets using the Buy/Sell system instead of Yes/No outcomes.
    - Saves the bet in the database under the AI user.
    """
    try:
        # AI bot user ID
        bot_user_id = "8AAjT02uf3XwQuHaebR4bTacbw92"

        # Fetch the event and match details
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise ValueError(f"Event with ID {event_id} not found.")

        match = db.query(Match).filter(Match.id == event.match_id).first()
        if not match:
            raise ValueError(f"Match associated with event ID {event_id} not found.")

        # Predict probabilities for the two teams
        prob_team1, prob_team2 = predict(match.team1, match.team2)

        # Log probabilities for debugging
        logging.info(f"Probabilities for event {event_id}: {match.team1} ({prob_team1}), {match.team2} ({prob_team2})")

        # Determine buy or sell action based on probabilities
        if prob_team1 > prob_team2:
            selected_team = match.team1
            bet_type = "buy"
        else:
            selected_team = match.team2
            bet_type = "sell"

        # Adjust bet size based on probability difference
        probability_diff = abs(prob_team1 - prob_team2)
        base_bet = randint(1, 10)  # Base number of shares to bet
        if probability_diff > 0.8:
            bet_size = base_bet * 0.5  # Smaller bet to avoid bias
        elif probability_diff > 0.6:
            bet_size = base_bet * 0.75
        else:
            bet_size = base_bet  # Normal bet for close probabilities

        # Get the latest buy/sell prices from event variations
        market_prices = get_share_price(event_id, db)
        market_price = market_prices["buy_price"] if bet_type == "buy" else market_prices["sell_price"]

        # Prepare the buy/sell request payload
        trade_request = BuyShareRequest(
            user_id=bot_user_id,
            event_id=event_id,
            bet_type=bet_type,
            shareCount=int(bet_size),
            share_price=market_price
        )

        # Call the trade API to place the bet
        trade_response = buy_share(trade_request, db)
        logging.info(f"AI bot placed {bet_type} bet on {selected_team}: {trade_response}")

    except Exception as e:
        logging.error(f"Error in AI bot betting for event {event_id}: {str(e)}")


def fetch_and_store_matches(db: Session):
    """
    Fetch and store matches for the next day and create events.
    """
    load_dotenv(dotenv_path='app\.env')
    API_URL = "https://v1.basketball.api-sports.io/games"
    key = os.getenv('BASKETBALL_API_KEY')

    API_HEADERS = {
        "x-rapidapi-key": os.getenv('BASKETBALL_API_KEY'),
        "x-rapidapi-host": "v1.basketball.api-sports.io"
    }
    LEAGUES = [12, 116]
    next_day = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    for league_id in LEAGUES:
        payload = {
            "league": league_id,
            "season": "2024-2025",
            "date": next_day
        }
        response = requests.get(API_URL, headers=API_HEADERS, params=payload)
        if response.status_code == 200:
            match_data = response.json().get("response", [])
            for match in match_data:
                existing_match = db.query(Match).filter_by(id=match["id"]).first()
                if not existing_match:
                    match_start = datetime.fromtimestamp(match["timestamp"])
                    new_match = Match(
                        id=match["id"],
                        sport="basketball",
                        league=match["league"]["name"],
                        team1=match["teams"]["home"]["name"],
                        team2=match["teams"]["away"]["name"],
                        match_time=match_start,
                        bet_end_time=match_start + timedelta(hours=2), 
                    )
                    db.add(new_match)
                    db.commit()  # Commit the match before creating events
                    add_events_for_match(new_match.id, db)
                else:
                    print(f"Match {match['id']} already exists.")
        else:
            print(f"Failed to fetch matches for league {league_id}: {response.status_code}")

    return (f"Matches and events for {next_day} processed.")


def add_team_points_events(match_id: int, team_name: str, db: Session):
    """
    Create over/under events for a team's points in a match.
    """
    thresholds = [80, 90, 100, 110, 120]  # Adjust if needed
    initial_buy_price = 51
    initial_sell_price = 49
    events = []

    for threshold in thresholds:
        events.append(Event(
            match_id=match_id,
            type=EventType.OVER_UNDER,
            heading=f"{team_name} Points - Over/Under Indices",
            question=f"Will {team_name} score over {threshold} points?",
            threshold=threshold,
            buy_sell_index=100,
            buy_price=initial_buy_price,
            sell_price=initial_sell_price,
            variations=[{"timestamp": datetime.now().isoformat(), "buy_price": initial_buy_price, "sell_price": initial_sell_price}]
        ))

    db.add_all(events)
    db.commit()
    print(f"Team points events added for {team_name} in match ID {match_id}.")


def add_events_for_match(match_id: int, db: Session):
    """
    Add predefined events for a match including match result, handicap, and total points.
    """
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found.")

    events = []
    initial_buy_price = 51  # Set initial buy price
    initial_sell_price = 49  # Set initial sell price
    spread_factor = 2

    ### 🔹 MATCH RESULT EVENTS ###
    events.append(Event(
        match_id=match.id,
        type=EventType.MATCH_RESULT,
        heading="Match Result - 100 Indices",
        question=f"{match.team1} to Win?",
        buy_sell_index=100,
        buy_price=initial_buy_price,
        sell_price=initial_sell_price,
        variations=[{"timestamp": datetime.now().isoformat(), "buy_price": initial_buy_price, "sell_price": initial_sell_price}]
    ))
    
    events.append(Event(
        match_id=match.id,
        type=EventType.MATCH_RESULT,
        heading="Match Result - 100 Indices",
        question=f"{match.team2} to Win?",
        buy_sell_index=100,
        buy_price=initial_buy_price,
        sell_price=initial_sell_price,
        variations=[{"timestamp": datetime.now().isoformat(), "buy_price": initial_buy_price, "sell_price": initial_sell_price}]
    ))

    ### 🔹 HANDICAP BETTING ###
    handicap_values = [-7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5]  # Example spread values
    for handicap in handicap_values:
        events.append(Event(
            match_id=match.id,
            type=EventType.HANDICAP,
            heading="Handicap - 100 Indices",
            question=f"{match.team1} {handicap}?",
            threshold=handicap,
            buy_sell_index=100,
            buy_price=initial_buy_price,
            sell_price=initial_sell_price,
            variations=[{"timestamp": datetime.now().isoformat(), "buy_price": initial_buy_price, "sell_price": initial_sell_price}]
        ))
        events.append(Event(
            match_id=match.id,
            type=EventType.HANDICAP,
            heading="Handicap - 100 Indices",
            question=f"{match.team2} {-handicap}?",
            threshold=-handicap,
            buy_sell_index=100,
            buy_price=initial_buy_price,
            sell_price=initial_sell_price,
            variations=[{"timestamp": datetime.now().isoformat(), "buy_price": initial_buy_price, "sell_price": initial_sell_price}]
        ))

    ### 🔹 TOTAL POINTS BETTING ###
    total_points_thresholds = [210.5, 214.5, 218.5, 222.5, 226.5]  # Example Over/Under Values
    for threshold in total_points_thresholds:
        events.append(Event(
            match_id=match.id,
            type=EventType.OVER_UNDER,
            heading="Total Points - Over/Under Indices",
            question=f"Will the total points be over {threshold}?",
            threshold=threshold,
            buy_sell_index=100,
            buy_price=initial_buy_price,
            sell_price=initial_sell_price,
            variations=[{"timestamp": datetime.now().isoformat(), "buy_price": initial_buy_price, "sell_price": initial_sell_price}]
        ))

    db.add_all(events)
    db.commit()

    # Add team-specific events (Each Team’s Total Points)
    add_team_points_events(match.id, match.team1, db)
    add_team_points_events(match.id, match.team2, db)

    print(f"Events added for match ID {match_id}.")


def update_match_scores_and_status(db: Session):
    """
    Update scores for matches that have started but are not finished.
    - Updates quarter-wise scores and total points.
    - Sets `bet_end_time` when Q4 starts.
    - Updates match status when match ends.
    """
    load_dotenv(dotenv_path='app\.env')
    API_URL = "https://v1.basketball.api-sports.io/games"
    API_HEADERS = {
        "x-rapidapi-key": os.getenv('BASKETBALL_API_KEY'),
        "x-rapidapi-host": "v1.basketball.api-sports.io"
    }

    current_time = datetime.utcnow()

    # ✅ Only fetch matches where the current time > match start time and status is not "finished"
    ongoing_matches = db.query(Match).filter(
        Match.match_time <= current_time,  # Match has started
        Match.status != "finished"  # Match is not finished
    ).all()

    for match in ongoing_matches:
        response = requests.get(API_URL, headers=API_HEADERS, params={"id": match.id})
        if response.status_code != 200:
            logging.error(f"Failed to fetch match data for {match.id}")
            continue

        match_data = response.json()["response"][0]
        home_team = match_data["teams"]["home"]["name"]
        away_team = match_data["teams"]["away"]["name"]
        scores = match_data["scores"]

        # ✅ Update quarter-wise scores
        match.result = {
            "quarters": {
                "Q1": {home_team: scores["home"]["quarter_1"], away_team: scores["away"]["quarter_1"]},
                "Q2": {home_team: scores["home"]["quarter_2"], away_team: scores["away"]["quarter_2"]},
                "Q3": {home_team: scores["home"]["quarter_3"], away_team: scores["away"]["quarter_3"]},
                "Q4": {home_team: scores["home"]["quarter_4"], away_team: scores["away"]["quarter_4"]}
            },
            "total": {
                home_team: scores["home"]["total"],
                away_team: scores["away"]["total"]
            }
        }
        match.total_points = scores["home"]["total"] + scores["away"]["total"]

        # ✅ Set `bet_end_time` when Q4 starts
        if match_data["status"]["short"] == "Q4" and not match.bet_end_time:
            match.bet_end_time = datetime.utcnow()

        # ✅ Update match status when it ends
        if match_data["status"]["short"] in ["FT", "AOT"]:
            match.status = "finished"
            match.result_time = datetime.utcnow()

        db.add(match)

    db.commit()
    return {"message": "Match scores and status updated successfully"}

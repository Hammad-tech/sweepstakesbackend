from sqlalchemy.orm import Session  # Import Session
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time as t
from typing import List
import pytz
from .models import Match, Event, Share, User
import os
import requests
from sqlalchemy import func
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from datetime import timedelta
from fastapi import HTTPException


def scrape_and_store_matches(db: Session):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )

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
            local_tz = pytz.timezone("UTC")  # Use UTC timezone (or adjust as needed)
            local_time = local_tz.localize(local_time)

            match = Match(
                team1=team_names[0],
                team2=team_names[1],
                match_time=local_time,
                bet_start_time=local_time - timedelta(hours=5),
                bet_end_time=local_time + timedelta(minutes=75),
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
                variations=[],  # Add any variations if needed
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


def calculate_share_price(event_id: int, bet_type: str, db: Session):
    """
    Calculate the share price for a given event and bet type.

    Args:
        event_id (int): The ID of the event.
        bet_type (str): The type of bet ('buy' or 'sell').
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the share prices for 'yes' and 'no'.
    """
    # Validate the bet type parameter
    if bet_type not in ["buy", "sell"]:
        raise HTTPException(
            status_code=400, detail="Invalid bet type. Must be 'buy' or 'sell'."
        )

    # Retrieve the event details
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found.")

    # Calculate the total shares for both outcomes
    total_shares = event.total_yes_bets + event.total_no_bets

    # Set a fixed spread value
    spread_value = 2  # Fixed spread value

    # Avoid division by zero if no shares exist
    if total_shares == 0:
        yes_price = 50  # Base price
        no_price = 50
    else:
        # Calculate share prices as percentages
        yes_price = (event.total_yes_bets / total_shares) * 100
        no_price = (event.total_no_bets / total_shares) * 100

    # Adjust prices based on bet type
    if bet_type == "buy":
        yes_price += spread_value
        no_price += spread_value
    elif bet_type == "sell":
        yes_price -= spread_value
        no_price -= spread_value

    # Ensure prices don't fall below 0
    yes_price = max(0, yes_price)
    no_price = max(0, no_price)

    return {
        "event_id": event_id,
        "bet_type": bet_type,
        "yes_price": round(yes_price, 2),
        "no_price": round(no_price, 2),
    }


# Helper function: Calculate similarity
def string_similarity(a, b):
    """
    Calculate similarity between two strings using SequenceMatcher.
    Returns a float between 0 and 1.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Helper function: Find best match using similarity
def find_best_match(team_name, candidates):
    """
    Find the best match for team_name among candidates using similarity.
    Returns the best match and its similarity score.
    """
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        score = string_similarity(team_name, candidate)
        if score > best_score:
            best_match = candidate
            best_score = score
    return best_match, best_score


# Main function: Get match score
def get_match_score(team1, team2):
    """
    Fetch match score using Google Custom Search and scrape the result page.
    Returns 1 if team1 wins, -1 if team2 wins, and 0 for a tie.
    """
    # Load API key and CSE ID from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")  # Set this in your environment
    cse_id = os.getenv("CSE_ID")  # Set this in your environment

    if not api_key or not cse_id:
        return "Error: Missing API credentials."

    # Construct Google Custom Search query
    query = f"{team1} vs {team2} site:sofascore.com"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={cse_id}&key={api_key}"

    # Send request to Google API
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if "items" in results:
            # Get the first match URL
            match_url = results["items"][0]["link"]
            return scrape_match_score(match_url, team1, team2)
        else:
            return "Error: No results found for this match."
    elif response.status_code == 403:
        return "Error: API quota exceeded or invalid API key."
    else:
        return f"Error: HTTP {response.status_code}"


# Helper function: Scrape match score
def scrape_match_score(match_url, team1, team2):
    """
    Scrape the match page for scores and determine the winner.
    """
    response = requests.get(match_url)
    if response.status_code != 200:
        return "Error: Unable to fetch match page."

    # Parse the match page
    soup = BeautifulSoup(response.text, "html.parser")
    left_team = soup.find("div", {"data-testid": "left_team"})
    right_team = soup.find("div", {"data-testid": "right_team"})

    if left_team and right_team:
        # Extract team names
        left_team_name = left_team.find("bdi").text.strip()
        right_team_name = right_team.find("bdi").text.strip()

        # Extract scores
        left_team_score = soup.find("span", {"data-testid": "left_score"}).text.strip()
        right_team_score = soup.find(
            "span", {"data-testid": "right_score"}
        ).text.strip()

        try:
            left_team_score = int(left_team_score)
            right_team_score = int(right_team_score)
        except ValueError:
            return "Error: Unable to parse scores as integers."

        # Match team names using similarity
        teams = [left_team_name, right_team_name]
        match1, score1 = find_best_match(team1, teams)
        match2, score2 = find_best_match(team2, teams)

        # Determine the winner
        if match1 == left_team_name:
            match_team1_score = left_team_score
        else:
            match_team1_score = right_team_score

        if match2 == right_team_name:
            match_team2_score = right_team_score
        else:
            match_team2_score = left_team_score

        if match_team1_score > match_team2_score:
            return 1  # Team1 wins
        elif match_team2_score > match_team1_score:
            return -1  # Team2 wins
        else:
            return 0  # Tie
    else:
        return "Error: Unable to find team information on the page."


def calculate_results_for_event(event_id: int, db: Session):
    # Fetch the event
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise ValueError(f"Event with ID {event_id} not found.")

    # Fetch the match associated with the event
    match = db.query(Match).filter(Match.id == event.match_id).first()
    if not match:
        raise ValueError(f"Match associated with event ID {event_id} not found.")

    # Check if the match start time is at least 3 hours before
    if func.now() < match.match_time + timedelta(hours=3):
        raise ValueError("Results cannot be calculated as the match has not ended.")

    # Determine the winner
    match_result = get_match_score(match.team1, match.team2)
    if match_result not in [1, -1, 0]:
        raise ValueError("Invalid result from get_match_score function.")

    # 1 = Team 1 wins, -1 = Team 2 wins, 0 = Draw
    winning_outcome = (
        "yes" if match_result == 1 else "no" if match_result == -1 else "draw"
    )

    # Fetch all shares for the event
    shares = db.query(Share).filter(Share.event_id == event_id).all()

    for share in shares:
        # Skip processing for draw as no bets are resolved
        if winning_outcome == "draw":
            continue

        # Calculate profit/loss for the user
        user = db.query(User).filter(User.id == share.user_id).first()
        if not user:
            continue

        if share.outcome == winning_outcome:
            # Winning bet
            if share.bet_type == "buy":
                profit = share.amount * (100 - share.share_price) / 100
                user.sweeps_points += profit
            elif share.bet_type == "sell":
                profit = share.amount * (share.share_price) / 100
                user.sweeps_points += profit
        else:
            # Losing bet
            if share.bet_type == "buy":
                loss = share.amount * (share.share_price) / 100
                user.sweeps_points -= loss
            elif share.bet_type == "sell":
                loss = share.amount * (100 - share.share_price) / 100
                user.sweeps_points -= loss

        # Remove resolved shares
        db.delete(share)

    # Commit changes
    db.commit()
    return {"message": "Results calculated successfully for the event."}

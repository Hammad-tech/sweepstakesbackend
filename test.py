from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Setup the WebDriver (assuming Chrome here)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


# URL to scrape
url = "https://www.pinnacle.com/en/basketball/matchups/"

# Open the URL
driver.get(url)

# Allow time for the page to load and dynamic content to appear
time.sleep(5)

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

    # Scroll the table down incrementally
    driver.execute_script(
        "arguments[0].scrollTop += arguments[0].offsetHeight", scrollable_table
    )
    time.sleep(2)  # Allow time for the content to load

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

# Print the scraped data
for entry in data:
    print(entry)


print(len(data))

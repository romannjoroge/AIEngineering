# Functions to scrape data from wikipedia
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import random
import numpy as np
import pandas as pd
import math, time
from datetime import datetime
from bs4 import BeautifulSoup
import re

class ScrapingException(BaseException):
    """Errors gotten when trying to scrape from wikipedia"""

# Use different user agents to mimic different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

def get_session():
    """
    Function to return a session that I can use to make requests to wikipedia's pages
    """
    session = requests.Session()
    
    # Add retry logic
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter=adapter)
    return session
    
SESSION = get_session()

def get_wikipedia_page(url):
    """
    A function to try getting a wikipedia page
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    # Make a request to Wikipedia
    try:
        print("Getting response")
        wikipedia_response = SESSION.get(url, headers=headers, timeout=15)
        if wikipedia_response.status_code != 200:
            raise ScrapingException(f"Wikipedia refused to send page with status code {wikipedia_response.status_code}")
        else:
            return wikipedia_response
    except Exception as e:
        print(f"Could not get page because of {e}")
        raise Exception("Unknown error")
    
class PlayerDetails:
    def __init__(self, from_year: int | None, to_year: int | None, appearances: int | None, goals: int | None, team: str | None):
        self.from_year = from_year
        self.to_year = to_year
        self.appearances = appearances
        self.goals = goals
        self.team = team
        
    def __repr__(self):
        return f"Found data row with from: {self.from_year} to: {self.to_year} team: {self.team} appearances: {self.appearances} and goals: {self.goals}"

def extract_infobox(player_wikipedia: str, player_id: str, problematic_players: list[str]) -> list[PlayerDetails] | None:
    """
    Extract player details from a player's wikipedia page
    
    Args:
        player_wikipedia: URL of a player's wikipedia page
        player_id: id of player being processed
        problematic_players: list of players whose wikipedia page couldn't be parsed
        
    Returns:
        players: A list of player objects if response of player was parsable
    """
    try:    
        player_response = get_wikipedia_page(player_wikipedia)
        assert player_response.status_code == 200, "Can't parse an unsucessful response"
        
        # Parse HTML of the player's Wiki page
        soup = BeautifulSoup(player_response.text, 'html.parser')

        # Look for the players info box
        tables = soup.select_one("table.infobox")

        # Store unparsable player for later processing
        if not tables:
            print(f"Player does not have info box in wiki: {player_wikipedia}")
            problematic_players.append(player_id)
            return None

        player_details = []
        table_rows = tables.find_all("tr")
        current = 0
        section_name = ""
        content_num = 0

        # Scanning table rows
        while True:
            if current >= len(table_rows):
                print("Consumed all rows in table")
                break
            
            current_row = table_rows[current]
            
            # Checking if row is the heading of a section
            if len(current_row.find_all("td")) == 0:
                section_name = current_row.select_one("th").text
                content_num = 0
                
                # Check if it's the section we want
                if "Senior career" in section_name:
                    # While next item is content
                    while current + 1 < len(table_rows) and len(table_rows[current + 1].find_all("td")) > 0:
                        next_content = table_rows[current + 1]
                        # Check if it is a table row
                        if len(next_content.find_all("b")) > 0:
                            print(f"Found title row {next_content.find("b")}")
                        else:
                            # Work with data that looks like 1 th 3 td
                            data_timeline = next_content.find("th")
                            
                            from_year = None
                            to_year = None
                            team = None
                            appearances = None
                            goals = None
                            
                            # Extract from and to years
                            if data_timeline:
                                raw_timeline = data_timeline.get_text(strip=True)
                                if len(raw_timeline) > 4:
                                    from_year = int(re.sub(r'\D', '', raw_timeline[0:4]))
                                    if len(raw_timeline) > 5:
                                        to_year = int(re.sub(r'\D', '', raw_timeline[4:]))
                                else:
                                    from_year = int(re.sub(r'\D', '', raw_timeline))
                            
                            rest_data = next_content.find_all("td")
                            if len(rest_data) < 3:
                                print(f"Unkown rest data {rest_data}")
                                problematic_players.append(player_id)
                            # Get team, appearance, goals
                            else:
                                raw_appearance = rest_data[1].get_text(strip=True)
                                raw_goals = rest_data[2].get_text(strip=True)
                                team = rest_data[0].get_text(strip=True)
                                if len(raw_appearance) > 0 and raw_appearance != "?":
                                    appearances = int(re.sub(r'\D', '', raw_appearance))
                                if len(raw_goals) > 0 and raw_goals != "(?)":
                                    goals = int(re.sub(r'\D', '', raw_goals))
                        
                            print(f"Found data row with from: {from_year} to: {to_year} team: {team} appearances: {appearances} and goals: {goals}")
                            player_details.append(PlayerDetails(
                                from_year=from_year,
                                to_year=to_year,
                                appearances=appearances,
                                goals=goals,
                                team = team
                            ))
                            content_num += 1
                        
                        current += 1
                        
                    # Increment
                    current += 1
                    print(f"Section {section_name} had {content_num} items")
                else:
                    current += 1
            else:
                current += 1
                
        return player_details
    except Exception as e:
        print(f"Unexpected error {e}")
        problematic_players.append(player_id)
    except ScrapingException as e:
        print(f"Error getting Wikipedia page {e}")
        problematic_players.append(player_id)
        
class Player:
    def __init__(self, id: str, wikipedia_url: str, problematic_players: list[str]):
        self.id = id
        self.wikipedia_url = wikipedia_url
        self.wikipedia_details = self._extract_wikipedia_details(problematic_players=problematic_players)
        
    def _extract_wikipedia_details(self, problematic_players: list[str]) -> list[PlayerDetails] | None:
        """
        Returns player details that have been extracted from their wikipedia page
        
        Args:
            problematic_players : A list used to store all player ids that had an issue extracting details from
            
        Returns:
            player_details: A list of player details from wikipedia
        """
        return extract_infobox(self.wikipedia_url, self.id, problematic_players)
    
    def to_csv(self) -> list[str]:
        """ 
        Returns strings that will represent the player in a CSV file
        
        Returns:
            entries (str): Entries in CSV file
        """
        entries = []
        if self.wikipedia_details:
            for detail in self.wikipedia_details:
                entries.append(f"{self.id},{detail.appearances},{detail.team},{detail.goals},{detail.from_year},{detail.to_year}\n")
            
        return entries
    
    def __repr__(self):
        return f"Player: {self.wikipedia_details}"
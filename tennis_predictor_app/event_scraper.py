import asyncio
import datetime
import csv
from playwright.async_api import async_playwright

# List of words to filter out from tournament names/categories.
# Added "wta" to the list. Case-insensitive.
FILTER_KEYWORDS = ["itf", "utr", "wta", "wheelchairs", "girls", "junior", "challenger"]

async def get_and_write_matches(date_str: str, csv_writer):
    """
    Uses Playwright to fetch match data, filters for non-ITF/UTR/WTA singles matches,
    and writes the results to a CSV file.
    """
    print(f"\n--- Scraping Matches for {date_str} ---")

    matches_url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date_str}"

    # JavaScript to execute in the browser to fetch API data
    js_to_fetch_data = f'''
        async () => {{
            try {{
                const response = await fetch('{matches_url}');
                if (!response.ok) {{
                    return {{'error': `API responded with status: ${{response.status}}`}};
                }}
                return await response.json();
            }} catch (e) {{
                return {{'error': e.toString()}};
            }}
        }}
    '''

    match_data = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            print("Initializing browser session...")
            await page.goto("https://www.sofascore.com/tennis", wait_until="domcontentloaded")
            print("Session established. Fetching data...")

            match_data = await page.evaluate(js_to_fetch_data)

            await browser.close()
            print("Browser closed.")

    except Exception as e:
        print(f"[!!] An error occurred during Playwright execution: {e}")
        return

    if not match_data or match_data.get('error'):
        print(f"[!!] FAILED to fetch matches for {date_str}.")
        if match_data:
            print(f"    Error from browser fetch: {match_data.get('error')}")
        return

    events = match_data.get('events', [])
    if not events:
        print("No matches found for this date.")
        return

    print(f"Found {len(events)} total events. Filtering and writing to CSV...")

    count_written = 0
    for event in events:
        category = event.get('tournament', {}).get('category', {}).get('name', 'N/A')
        tournament_name = event.get('tournament', {}).get('name', 'N/A')
        home_team = event.get('homeTeam', {}).get('name', 'N/A')
        away_team = event.get('awayTeam', {}).get('name', 'N/A')

        # Combine category and tournament name for easier filtering
        full_tournament_info = f"{category} {tournament_name}".lower()

        # --- NEW & IMPROVED FILTERING LOGIC ---
        # 1. Skip if the combined info contains any filter keyword (ITF, UTR, WTA)
        if any(keyword in full_tournament_info for keyword in FILTER_KEYWORDS):
            continue

        # 2. Skip if it's a doubles match
        if "doubles" in full_tournament_info or "/" in home_team or "/" in away_team:
            continue

        # If the event passed all filters, extract its data
        start_timestamp = event.get('startTimestamp')
        match_time = datetime.datetime.fromtimestamp(start_timestamp).strftime('%H:%M')

        # Write the filtered row to the CSV
        csv_writer.writerow([
            date_str,
            match_time,
            category,
            tournament_name,
            home_team,
            away_team
        ])
        count_written += 1

    print(f"Wrote {count_written} filtered matches to the CSV file.")

async def scrape_and_save_matches():
    """Scrapes and saves upcoming matches to a CSV file."""
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)

    today_str = today.strftime('%Y-%m-%d')
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')

    output_filename = "sofascore_filtered_matches.csv"

    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Time', 'Category', 'Tournament', 'Player 1', 'Player 2'])

        await get_and_write_matches(today_str, writer)
        await get_and_write_matches(tomorrow_str, writer)

    print(f"\n✅ All done! Data saved to '{output_filename}'")
    return output_filename
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from fastapi.middleware.cors import CORSMiddleware
from .scraping import scrape_and_store_matches
from .db_config import get_db

# FastAPI app configuration
def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: run_scrape_and_store_matches(), "cron", hour=12, minute=10)
    scheduler.start()

def run_scrape_and_store_matches():
    # Get a session from the DB
    db = next(get_db())
    scrape_and_store_matches(db)
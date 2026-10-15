import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
PAGES = ["Plan a journey", "Confirmation chances", "Train suggestions", "Delays and disruptions", "Fare forecast",
         "Support chat", "Fraud and bots", "Watchtower governance"]


@pytest.fixture(scope="module", autouse=True)
def env(tmp_path_factory):
    os.environ["TRAINWISE_ARTIFACT_DIR"] = str(tmp_path_factory.mktemp("dash_artifacts"))
    os.environ["TRAINWISE_TODAY"] = "2026-10-15"
    from trainwise import platform as plat
    plat._PLATFORM = None


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, label


def test_every_page_renders_and_main_actions_work():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    actions = {"Plan a journey": ["Plan my journey", "Approve and book"], "Confirmation chances": ["Predict"],
               "Train suggestions": ["Suggest trains"], "Delays and disruptions": ["Forecast delay"],
               "Fare forecast": ["Forecast fare"], "Fraud and bots": ["Score session"],
               "Watchtower governance": ["Send normal traffic", "Inject selected failures", "Run governance cycle"]}
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        for label in actions.get(page, []):
            click(at, label)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "credit_app.json"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def normalize_phone(phone: str) -> str:
    return "".join(char for char in phone if char.isdigit())


def build_default_data() -> dict:
    return {
        "users": [
            {
                "id": 1,
                "username": "admin",
                "email": "admin@example.com",
                "phone": "9876543210",
                "password_hash": hash_password("admin123"),
            }
        ],
        "predictions": [],
    }


def load_db() -> dict:
    DATA_DIR.mkdir(exist_ok=True)

    if not DB_PATH.exists():
        data = build_default_data()
        DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    data = json.loads(DB_PATH.read_text(encoding="utf-8"))

    for user in data.get("users", []):
        user.setdefault("email", f"{user['username']}@example.com")
        user.setdefault("phone", "")
        if user.get("username") == "admin" and not user.get("phone"):
            user["phone"] = "9876543210"
        if user.get("username") == "admin" and not user.get("email"):
            user["email"] = "admin@example.com"

    data.setdefault("predictions", [])
    save_db(data)
    return data


def save_db(data: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def init_db() -> None:
    load_db()


def create_user(username: str, email: str, phone: str, password: str) -> bool:
    data = load_db()
    clean_username = username.strip().lower()
    clean_email = email.strip().lower()
    clean_phone = normalize_phone(phone)

    if any(user["username"] == clean_username for user in data["users"]):
        return False
    if any(user.get("email", "").lower() == clean_email for user in data["users"]):
        return False
    if clean_phone and any(user.get("phone", "") == clean_phone for user in data["users"]):
        return False

    next_id = max((user["id"] for user in data["users"]), default=0) + 1
    data["users"].append(
        {
            "id": next_id,
            "username": clean_username,
            "email": clean_email,
            "phone": clean_phone,
            "password_hash": hash_password(password),
        }
    )
    save_db(data)
    return True


def verify_user(identifier: str, password: str, phone: str = ""):
    data = load_db()
    clean_identifier = identifier.strip().lower()
    clean_phone = normalize_phone(phone)
    password_hash = hash_password(password)

    for user in data["users"]:
        identifier_match = clean_identifier in {
            user["username"],
            user.get("email", "").lower(),
        }
        phone_match = not clean_phone or user.get("phone", "") == clean_phone

        if identifier_match and phone_match and user["password_hash"] == password_hash:
            return {"id": user["id"], "username": user["username"]}

    return None


def get_user_by_contact(identifier: str, contact_type: str):
    data = load_db()
    clean_identifier = identifier.strip().lower()

    for user in data["users"]:
        if contact_type == "email" and user.get("email", "").lower() == clean_identifier:
            return {"id": user["id"], "username": user["username"]}
        if contact_type == "phone" and user.get("phone", "") == normalize_phone(identifier):
            return {"id": user["id"], "username": user["username"]}

    return None


def update_user_password(user_id: int, new_password: str) -> bool:
    data = load_db()

    for user in data["users"]:
        if user["id"] == user_id:
            user["password_hash"] = hash_password(new_password)
            save_db(data)
            return True

    return False


def save_prediction(
    user_id: int,
    age: int,
    sex: str,
    job: int,
    housing: str,
    saving_accounts: str,
    checking_account: str,
    credit_amount: float,
    duration: int,
    purpose: str,
    result: str,
    confidence: float,
    risk_band: str,
) -> None:
    data = load_db()
    next_id = max((item["id"] for item in data["predictions"]), default=0) + 1

    data["predictions"].append(
        {
            "id": next_id,
            "user_id": user_id,
            "Age": age,
            "Sex": sex,
            "Job": job,
            "Housing": housing,
            "Saving Accounts": saving_accounts,
            "Checking Account": checking_account,
            "Credit Amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose,
            "Result": result,
            "Confidence": confidence,
            "Risk Band": risk_band,
        }
    )
    save_db(data)


def get_prediction_history(user_id: int) -> pd.DataFrame:
    data = load_db()
    rows = [item for item in reversed(data["predictions"]) if item["user_id"] == user_id]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop(columns=["id", "user_id"])

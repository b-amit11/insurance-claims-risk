import os
import numpy as np
import pandas as pd

# Random generator for reproducibility
rng = np.random.default_rng(42)

# How many synthetic claims to generate
N_CLAIMS = 8000

# Where the cleaned CSV will be saved
OUTPUT_PATH = os.path.join("data", "claims_clean.csv")


def generate_synthetic_claims(n_claims: int) -> pd.DataFrame:
    """
    Generate a synthetic insurance claims dataset.

    Each row represents a claim with:
    - Region, State
    - ClaimCategory (Auto, Home, Health, Commercial)
    - Severity (Low, Medium, High)
    - IncidentDate, ReportDate, CloseDate
    - WrittenPremium, ClaimAmount, PaidAmount, ReserveAmount
    - FraudFlag, IsClosed
    """

    # Random incident dates between 2022-01-01 and 2024-12-30
    incident_dates = pd.to_datetime(
        rng.choice(
            pd.date_range("2022-01-01", "2024-12-30", freq="D"),
            size=n_claims,
        )
    )

    # Region and state
    regions = ["Northeast", "Midwest", "South", "West"]
    region_probs = [0.25, 0.25, 0.30, 0.20]
    region = rng.choice(regions, p=region_probs, size=n_claims)

    region_state_map = {
        "Northeast": ["NY", "MA", "PA"],
        "Midwest": ["IL", "MI", "OH"],
        "South": ["TX", "FL", "GA"],
        "West": ["CA", "WA", "CO"],
    }
    state = [rng.choice(region_state_map[r]) for r in region]

    # Claim category and severity
    categories = ["Auto", "Home", "Health", "Commercial"]
    category_probs = [0.45, 0.25, 0.20, 0.10]
    claim_category = rng.choice(categories, p=category_probs, size=n_claims)

    severities = ["Low", "Medium", "High"]
    severity_probs = [0.6, 0.3, 0.1]
    severity = rng.choice(severities, p=severity_probs, size=n_claims)

    # IDs
    policy_ids = rng.integers(100000, 200000, size=n_claims)
    customer_ids = rng.integers(60000, 90000, size=n_claims)

    # Base premium by category
    base_premium = {
        "Auto": 800,
        "Home": 1200,
        "Health": 1500,
        "Commercial": 3000,
    }

    # Severity factor ranges (fraction of premium)
    severity_factor = {
        "Low": (0.05, 0.2),
        "Medium": (0.2, 0.7),
        "High": (0.7, 2.0),
    }

    written_premium = []
    claim_amount = []

    # For each claim, simulate premium and claim amount
    for cat, sev in zip(claim_category, severity):
        bp = base_premium[cat]

        # Premium: log-normal around the base premium
        premium = rng.lognormal(mean=np.log(bp), sigma=0.4)
        premium = np.clip(premium, bp * 0.5, bp * 2.5)
        written_premium.append(premium)

        # Claim amount: fraction of premium depending on severity
        low_pct, high_pct = severity_factor[sev]
        pct = rng.uniform(low_pct, high_pct)
        amount = premium * pct
        claim_amount.append(amount)

    # Report date = incident date + 0–30 days
    report_lag_days = rng.integers(0, 31, size=n_claims)
    report_dates = incident_dates + pd.to_timedelta(report_lag_days, unit="D")

    # Open/closed status and close dates
    is_closed = rng.choice([1, 0], p=[0.7, 0.3], size=n_claims)
    close_lag_days = rng.integers(5, 365, size=n_claims)
    close_dates = []
    for closed, inc_date, lag in zip(is_closed, incident_dates, close_lag_days):
        if closed:
            close_dates.append(inc_date + pd.to_timedelta(int(lag), unit="D"))
        else:
            close_dates.append(pd.NaT)
    close_dates = pd.to_datetime(close_dates)

    # Paid vs reserve amounts
    paid_amount = []
    reserve_amount = []
    for amt, closed in zip(claim_amount, is_closed):
        if closed:
            # Closed claims: mostly fully paid
            paid = amt * rng.uniform(0.8, 1.1)
            reserve = 0.0
        else:
            # Open claims: partial payment + reserve
            paid = amt * rng.uniform(0.2, 0.6)
            reserve = amt - paid
        paid_amount.append(paid)
        reserve_amount.append(reserve)

    # Small fraud percentage
    fraud_flag = rng.choice([0, 1], p=[0.95, 0.05], size=n_claims)

    # Seasonality by incident month
    incident_month = incident_dates.month
    seasonal_multiplier = []
    for cat, m in zip(claim_category, incident_month):
        mult = 1.0
        if cat == "Auto" and m in [12, 1, 2]:
            mult *= 1.3  # higher in winter
        if cat == "Home" and m in [6, 7, 8]:
            mult *= 1.25  # higher in summer
        seasonal_multiplier.append(mult)

    seasonal_multiplier = np.array(seasonal_multiplier)

    # Apply seasonality to amounts
    claim_amount = np.round(claim_amount * seasonal_multiplier, 2)
    paid_amount = np.round(paid_amount * seasonal_multiplier, 2)
    reserve_amount = np.clip(np.round(claim_amount - paid_amount, 2), 0, None)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "ClaimID": np.arange(1, n_claims + 1),
            "PolicyID": policy_ids,
            "CustomerID": customer_ids,
            "Region": region,
            "State": state,
            "ClaimCategory": claim_category,
            "Severity": severity,
            "IncidentDate": incident_dates,
            "ReportDate": report_dates,
            "CloseDate": close_dates,
            "IsClosed": is_closed,
            "WrittenPremium": np.round(written_premium, 2),
            "ClaimAmount": claim_amount,
            "PaidAmount": paid_amount,
            "ReserveAmount": reserve_amount,
            "FraudFlag": fraud_flag,
        }
    )

    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful time-based and ratio features for analysis and dashboards.
    """

    # Time-related features for trends / seasonality
    df["IncidentYear"] = df["IncidentDate"].dt.year
    df["IncidentMonth"] = df["IncidentDate"].dt.month
    df["IncidentMonthName"] = df["IncidentDate"].dt.month_name()
    df["IncidentQuarter"] = df["IncidentDate"].dt.to_period("Q").astype(str)

    # Claim duration (0 for open claims)
    df["ClaimDurationDays"] = (df["CloseDate"] - df["IncidentDate"]).dt.days
    df["ClaimDurationDays"] = df["ClaimDurationDays"].fillna(0)

    # Loss ratio (important insurance metric)
    df["LossRatio"] = df["ClaimAmount"] / df["WrittenPremium"]
    df["LossRatio"] = df["LossRatio"].replace([np.inf, -np.inf], np.nan)
    df["LossRatio"] = df["LossRatio"].fillna(0)

    # Make sure flags are 0/1 ints
    df["IsClosed"] = df["IsClosed"].astype(int)
    df["FraudFlag"] = df["FraudFlag"].astype(int)

    return df


def main():
    # Make sure data directory exists
    os.makedirs("data", exist_ok=True)

    print("Generating synthetic claims data...")
    df = generate_synthetic_claims(N_CLAIMS)

    print("Cleaning and engineering features...")
    df = clean_and_engineer_features(df)

    print(f"Generated {len(df)} claims.")
    print("Preview of data:")
    print(df.head())

    # Save to CSV for Power BI / analysis
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nClean claims dataset saved to: {OUTPUT_PATH}")


if __name__ =="__main__":
    main()

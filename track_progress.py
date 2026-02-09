import pandas as pd
import numpy as np
import os
import datetime

def load_plan(filepath):
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please use CSV or XLSX.")

def restructure_plan(plan_df, overspend_ratio):
    adjust_factor = 1 + overspend_ratio / 100
    new_plan = plan_df.copy()
    new_plan['SpendAmt'] = plan_df['SpendAmt'] * adjust_factor
    total_available = plan_df['SaveAmt'] + plan_df['InvestAmt']
    reduction = total_available * (overspend_ratio / 100)
    new_plan['SaveAmt'] = plan_df['SaveAmt'] - reduction * 0.6
    new_plan['InvestAmt'] = plan_df['InvestAmt'] - reduction * 0.4
    return new_plan

def save_file_safely(df, filename, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    try:
        if path.endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
        print(f"Saved: {path}")
    except PermissionError:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_path = os.path.join(folder, f"{timestamp}_{filename}")
        if path.endswith(".csv"):
            df.to_csv(alt_path, index=False)
        else:
            df.to_excel(alt_path, index=False)
        print(f"⚠️ File in use. Saved as: {alt_path}")

def track_progress(plan_df, months_to_track):
    total_months = plan_df["Month"].nunique()
    months_to_track = min(months_to_track, total_months)

    print(f"\nTracking progress for first {months_to_track} months...\n")

    results = []
    months = sorted(plan_df["Month"].unique())[:months_to_track]

    for m in months:
        month_df = plan_df[plan_df["Month"] == m]
        for idx, row in month_df.iterrows():
            print(f"Month {m}, Period {int(row['Period'])}: Planned Spend={row['SpendAmt']}, Save={row['SaveAmt']}, Invest={row['InvestAmt']}")
            actual_spend = float(input("Enter Actual Spend: "))
            actual_save = float(input("Enter Actual Save: "))
            actual_invest = float(input("Enter Actual Invest: "))

            results.append({
                "Month": m,
                "Period": row["Period"],
                "PlannedSpend": row["SpendAmt"],
                "ActualSpend": actual_spend,
                "SpendDeviation(%)": ((actual_spend - row["SpendAmt"]) / row["SpendAmt"]) * 100,
                "PlannedSave": row["SaveAmt"],
                "ActualSave": actual_save,
                "SaveDeviation(%)": ((actual_save - row["SaveAmt"]) / row["SaveAmt"]) * 100,
                "PlannedInvest": row["InvestAmt"],
                "ActualInvest": actual_invest,
                "InvestDeviation(%)": ((actual_invest - row["InvestAmt"]) / row["InvestAmt"]) * 100,
            })

    results_df = pd.DataFrame(results)

    avg_spend_dev = results_df["SpendDeviation(%)"].mean()
    avg_save_dev = results_df["SaveDeviation(%)"].mean()
    avg_invest_dev = results_df["InvestDeviation(%)"].mean()

    print("\nProgress Summary:")
    print(f"Average Spend Deviation: {avg_spend_dev:.2f}%")
    print(f"Average Save Deviation: {avg_save_dev:.2f}%")
    print(f"Average Invest Deviation: {avg_invest_dev:.2f}%")

    save_file_safely(results_df, "progress_report.xlsx")

    if avg_spend_dev > 10:
        print("\nWarning: Overspending detected (>10% deviation).")
        choice = input("Do you want to restructure the plan? (y/n): ").lower()
        if choice == 'y':
            print("\nRestructuring plan based on overspending...")
            remaining_plan = plan_df[~plan_df["Month"].isin(months)]
            new_plan = restructure_plan(remaining_plan, avg_spend_dev)
            save_file_safely(new_plan, "restructured_plan.xlsx")
            print("New plan generated for remaining months with adjusted allocations.")
        else:
            print("No restructuring applied.")
    else:
        print("\nSpending within acceptable limits. No restructuring needed.")

def main():
    print("Track Progress Module")
    plan_path = input("Enter existing plan file path (CSV/XLSX): ").strip().replace('"', '').replace("'", "")

    if not os.path.exists(plan_path):
        print("File not found. Check your path and try again.")
        return

    plan_df = load_plan(plan_path)
    if not {"Month", "Period", "SpendAmt", "SaveAmt", "InvestAmt"}.issubset(plan_df.columns):
        print("Error: Missing required columns in plan file.")
        return

    total_months = plan_df["Month"].nunique()
    print(f"\nYour plan has {total_months} months available.")
    months_to_track = int(input("Enter number of months for which you have actual records: "))

    track_progress(plan_df, months_to_track)

if __name__ == "__main__":
    main()

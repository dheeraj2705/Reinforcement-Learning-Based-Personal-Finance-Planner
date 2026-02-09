from backend.rl_wrapper import generate_schedule

# Example user inputs
monthly_income = 95000
avg_expense = 45000
plan_months = 10

# Generate the schedule
df = generate_schedule(
    monthly_income=monthly_income/2,
    avg_monthly_expense=avg_expense,
    plan_months=plan_months
)

# Show the result in console
print(df)

# Save to CSV for Excel
df.to_csv("outputs/schedule_plan.csv", index=False)
print("✅ Schedule plan saved to outputs/schedule_plan_3.csv")

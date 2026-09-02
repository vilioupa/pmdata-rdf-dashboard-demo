import os

# Import each parser from the scripts package (one module per data type).
from scripts.context_rdf import process_context_data
from scripts.wellness_rdf import process_wellness_data
# steps_rdf is intentionally not imported: activity.py produces the step graph too.
# from scripts.steps_rdf import process_steps_data
from scripts.sleep_rdf import process_sleep_data
from scripts.hr_rdf import process_heart_rate_data

# Activity parser (also covers steps). Make sure the module name matches your file.
from scripts.activity import process_activity_data

def main():
    print("=== STARTING RDF CONVERSION FOR ALL USERS ===")
    data_directory = 'data'
    users_to_process = []

    # 1. Discover participants dynamically: every sub-folder of data/ is one user.
    for item in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item)
        # Keep only directories (e.g. P02, P03, P04, ...), ignore stray files.
        if os.path.isdir(item_path):
            users_to_process.append(item)

    # Process users in a stable, predictable order.
    users_to_process.sort()
    print(f"[*] Found {len(users_to_process)} users: {users_to_process}\n")

    # 2. Run every parser for each user in turn (all enabled for this sample dataset).
    for user in users_to_process:
        print(f"========== PROCESSING DATA FOR: {user} ==========")

        process_context_data(user)
        process_wellness_data(user)
        process_sleep_data(user)
        process_heart_rate_data(user)
        process_activity_data(user)

        print(f"========== COMPLETED USER: {user} ==========\n")

    print("=== PROCESS COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
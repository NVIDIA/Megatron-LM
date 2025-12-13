import os
import sys
import json
import requests
import argparse
from datetime import datetime, timedelta

# Constants
GITHUB_API_URL = "https://api.github.com"
SCHEDULE_FILE = ".github/oncall_schedule.json"
TEAM_SLUG = "mcore-oncalls"
TARGET_WEEKS = 12

def get_headers():
    token = os.environ.get("GH_TOKEN")
    if not token:
        # Fallback to GITHUB_TOKEN if GH_TOKEN not set
        token = os.environ.get("GITHUB_TOKEN")
        
    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN not set")
        sys.exit(1)
        
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

def get_repo_info():
    """Returns (owner, repo) from GITHUB_REPOSITORY env var."""
    repo_env = os.environ.get("GITHUB_REPOSITORY")
    if not repo_env:
        print("Error: GITHUB_REPOSITORY environment variable not set")
        sys.exit(1)
    parts = repo_env.split("/")
    return parts[0], parts[1]

def get_team_members(org, team_slug):
    """Fetches members of the GitHub team."""
    url = f"{GITHUB_API_URL}/orgs/{org}/teams/{team_slug}/members"
    headers = get_headers()
    
    members = []
    page = 1
    while True:
        resp = requests.get(f"{url}?per_page=100&page={page}", headers=headers)
        if resp.status_code != 200:
            print(f"Error fetching team members: {resp.status_code} {resp.text}")
            sys.exit(1)
        
        data = resp.json()
        if not data:
            break
            
        members.extend([m['login'] for m in data])
        if len(data) < 100:
            break
        page += 1
        
    return members

def load_schedule():
    if not os.path.exists(SCHEDULE_FILE):
        return []
    try:
        with open(SCHEDULE_FILE, 'r') as f:
            data = json.load(f)
            # Normalize to list of dicts if it's a list of strings
            schedule = []
            for item in data:
                if isinstance(item, str):
                    schedule.append({"user": item, "date": "YYYY-MM-DD"})
                else:
                    schedule.append(item)
            return schedule
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_schedule(schedule):
    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(schedule, f, indent=4)
        f.write('\n') # trailing newline

def rotate_schedule(repo_owner, dry_run=False):
    schedule = load_schedule()
    print(f"Current schedule length: {len(schedule)}")
    
    # 1. Rotate (Remove past week)
    # Only if schedule is not empty.
    if schedule:
        removed = schedule.pop(0)
        print(f"Rotated out: {removed}")
    else:
        print("Schedule empty, nothing to rotate.")

    # 2. Replenish
    ensure_schedule_filled(schedule, repo_owner)
    
    if not dry_run:
        save_schedule(schedule)
        print("Schedule updated and saved.")
    else:
        print("Dry run: Schedule not saved.")
        print(json.dumps(schedule, indent=4))

def get_last_wednesday():
    today = datetime.utcnow().date()
    # Monday=0, Wednesday=2
    offset = (today.weekday() - 2) % 7
    return today - timedelta(days=offset)

def ensure_schedule_filled(schedule, repo_owner):
    """Appends users to schedule until it reaches TARGET_WEEKS."""
    members = get_team_members(repo_owner, TEAM_SLUG)
    if not members:
        print("Warning: No team members found in mcore-oncalls.")
        return

    members.sort() # Deterministic order
    
    while len(schedule) < TARGET_WEEKS:
        # Determine start date for the new entry
        if not schedule:
            # Start with the most recent Wednesday if list is empty
            next_date = get_last_wednesday()
            
            # Start with the first member alphabetically if list is empty
            next_user = members[0]
        else:
            last_entry = schedule[-1]
            last_user = last_entry['user']
            
            # Parse last date and add 7 days
            try:
                last_date = datetime.strptime(last_entry['date'], "%Y-%m-%d").date()
                next_date = last_date + timedelta(days=7)
            except ValueError:
                # Fallback if date is invalid/placeholder
                next_date = get_last_wednesday() + timedelta(days=7 * len(schedule))

            try:
                # Find index of last scheduled user in the team list
                if last_user in members:
                    last_idx = members.index(last_user)
                    next_idx = (last_idx + 1) % len(members)
                    next_user = members[next_idx]
                else:
                    # Last user not in team, just pick first member
                    next_user = members[0]
            except ValueError:
                next_user = members[0]
        
        new_entry = {"user": next_user, "date": next_date.strftime("%Y-%m-%d")}
        schedule.append(new_entry)
        print(f"Appended: {new_entry}")

def assign_reviewer(pr_number):
    schedule = load_schedule()
    if not schedule:
        print("Error: Schedule is empty. Cannot assign reviewer.")
        sys.exit(1)
        
    current_entry = schedule[0]
    current_oncall = current_entry['user']
    print(f"Current oncall: {current_oncall} (Since {current_entry['date']})")
    
    owner, repo = get_repo_info()
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers"
    
    data = {"reviewers": [current_oncall]}
    resp = requests.post(url, headers=get_headers(), json=data)
    
    if resp.status_code in [201, 200]:
        print(f"Successfully requested review from {current_oncall}")
    else:
        print(f"Failed to request review: {resp.status_code} {resp.text}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Manage Oncall Schedule")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Rotate command
    parser_rotate = subparsers.add_parser("rotate", help="Rotate the schedule (remove first, append new)")
    parser_rotate.add_argument("--dry-run", action="store_true", help="Do not save changes")

    # Fill command (just fill up to 12 without rotating - useful for init)
    parser_fill = subparsers.add_parser("fill", help="Fill the schedule to 12 weeks without rotating")
    
    # Assign command
    parser_assign = subparsers.add_parser("assign", help="Assign current oncall to PR")
    parser_assign.add_argument("--pr", type=int, required=True, help="PR number")

    args = parser.parse_args()
    
    owner, _ = get_repo_info()
    
    if args.command == "rotate":
        rotate_schedule(owner, dry_run=args.dry_run)
    elif args.command == "fill":
        schedule = load_schedule()
        ensure_schedule_filled(schedule, owner)
        save_schedule(schedule)
        print("Schedule filled and saved.")
    elif args.command == "assign":
        assign_reviewer(args.pr)

if __name__ == "__main__":
    main()


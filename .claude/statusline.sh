#!/usr/bin/env bash
# ~/.claude/statusline.sh

PAYLOAD=$(cat)
MODEL=$(echo "$PAYLOAD" | jq -r '.model.display_name // "Unknown Model"')
CONTEXT_PCT=$(echo "$PAYLOAD" | jq -r '.context_window.used_percentage // 0')
INPUT_TOKENS=$(echo "$PAYLOAD" | jq -r '.context_window.total_input_tokens // 0')

# --- NEW: BURN RATE CALCULATOR ---
NOW=$(date +%s)
STATE_FILE="/tmp/claude_session_start"

# If tokens are 0 (or very low), assume it's a new /clear or fresh start and reset the timer
if [ "$INPUT_TOKENS" -lt 100 ]; then
    echo "$NOW" > "$STATE_FILE"
fi

# Ensure state file exists
if [ ! -f "$STATE_FILE" ]; then
    echo "$NOW" > "$STATE_FILE"
fi

START_TIME=$(cat "$STATE_FILE")
ELAPSED_MINUTES=$(( (NOW - START_TIME) / 60 ))

# Calculate Tokens Per Minute (TPM)
if [ "$ELAPSED_MINUTES" -gt 0 ]; then
    TPM=$(( INPUT_TOKENS / ELAPSED_MINUTES ))
else
    TPM=$INPUT_TOKENS
fi

# Format TPM for readability (e.g., 2500 -> 2.5k)
if [ "$TPM" -ge 1000 ]; then
    TPM_FORMATTED="$((TPM / 1000)).$(((TPM % 1000) / 100))k/min"
else
    TPM_FORMATTED="${TPM}/min"
fi

# ----  ROLLING-WINDOW BUDGET TRACKER (10 MIN WINDOW) ---
NOW=$(date +%s)
HISTORY_FILE="/tmp/claude_budget_history"
FIVE_HR_PCT=$(echo "$PAYLOAD" | jq -r '.rate_limits.five_hour.used_percentage // 0')

# 1. Append the current timestamp and percentage to our history log
echo "$NOW $FIVE_HR_PCT" >> "$HISTORY_FILE"

# 2. Prune the log: Only keep entries from the last 10 minutes (600 seconds)
CUTOFF=$(( NOW - 600 ))
if [ -f "$HISTORY_FILE" ]; then
    # Filter the file in place to keep it lightweight
    awk -v cutoff="$CUTOFF" '$1 >= cutoff' "$HISTORY_FILE" > "${HISTORY_FILE}.tmp" && mv "${HISTORY_FILE}.tmp" "$HISTORY_FILE"
fi

# 3. Extract the oldest and newest data points in our 10-minute window
OLDEST_ENTRY=$(head -n 1 "$HISTORY_FILE")
NEWEST_ENTRY=$(tail -n 1 "$HISTORY_FILE")

read OLD_TIME OLD_PCT <<< "$OLDEST_ENTRY"
read NEW_TIME NEW_PCT <<< "$NEWEST_ENTRY"

# 4. Compute the rolling metrics using awk for decimal precision
BUDGET_PREDICTION=$(awk -v old_t="$OLD_TIME" -v new_t="$NEW_TIME" -v old_p="$OLD_PCT" -v new_p="$NEW_PCT" -v now="$NOW" '
BEGIN {
    time_delta_mins = (new_t - old_t) / 60;
    pct_delta = new_p - old_p;

    # If a reset happened natively behind the scenes, pct_delta will be negative
    if (pct_delta < 0) pct_delta = 0;

    # Avoid divide by zero if we dont have enough history yet
    if (time_delta_mins > 0.1) {
        burn_rate_per_min = pct_delta / time_delta_mins;
    } else {
        burn_rate_per_min = 0;
    }

    # If you are actively burning budget (more than 0.01% per minute)
    if (burn_rate_per_min > 0.01) {
        remaining_pct = 100 - new_p;
        mins_left = remaining_pct / burn_rate_per_min;

        # We pass the minutes left out to a system command via a specific string format
        printf "RUN_DATE_CMD:%.0f", mins_left;
    } else {
        printf "💸 Safe";
    }
}')

# 5. If awk calculated an active burn, convert minutes left into a real clock time
if [[ "$BUDGET_PREDICTION" == RUN_DATE_CMD:* ]]; then
    MINUTES_LEFT=${BUDGET_PREDICTION#RUN_DATE_CMD:}
    EXHAUSTION_TIME=$(date -d "+${MINUTES_LEFT} minutes" +%I:%M%p)
    BUDGET_PREDICTION="💸 Overage @ ${EXHAUSTION_TIME}"
fi
# ----------------------------------------------------------

# Format total tokens for readability
if [ "$INPUT_TOKENS" -ge 1000 ]; then
    TOKENS_FORMATTED="$((INPUT_TOKENS / 1000)).$(((INPUT_TOKENS % 1000) / 100))k"
else
    TOKENS_FORMATTED="$INPUT_TOKENS"
fi

# Apply traffic-light colors
if [ "$CONTEXT_PCT" -lt 60 ]; then
    COLOR="\033[32m" # Green
    ICON="🟢"
elif [ "$CONTEXT_PCT" -lt 80 ]; then
    COLOR="\033[33m" # Yellow
    ICON="🟡"
else
    COLOR="\033[31m" # Red
    ICON="🔴"
fi
RESET="\033[0m"

# --- NEW: TEAM PLAN BUFFERS & TIMERS ---
FIVE_HR_PCT=$(echo "$PAYLOAD" | jq -r '.rate_limits.five_hour.used_percentage // 0')
SEVEN_DAY_PCT=$(echo "$PAYLOAD" | jq -r '.rate_limits.seven_day.used_percentage // 0')

# Extract the raw UNIX timestamps
FIVE_HR_RESET_UNIX=$(echo "$PAYLOAD" | jq -r '.rate_limits.five_hour.resets_at // empty')
SEVEN_DAY_RESET_UNIX=$(echo "$PAYLOAD" | jq -r '.rate_limits.seven_day.resets_at // empty')

# Format the 5-hour reset time (e.g., 04:30PM)
if [ -n "$FIVE_HR_RESET_UNIX" ]; then
    # The @ symbol tells Ubuntu date to read it as a UNIX epoch
    FIVE_HR_TIME=$(date -d @"$FIVE_HR_RESET_UNIX" +%I:%M%p)
    FIVE_HR_STR="🕒 5h: ${FIVE_HR_PCT}% (resets @ ${FIVE_HR_TIME})"
else
    FIVE_HR_STR="🕒 5h: ${FIVE_HR_PCT}%"
fi

# Format the 7-day reset time (e.g., Wed 04:30PM)
if [ -n "$SEVEN_DAY_RESET_UNIX" ]; then
    # Adding %a gives us the abbreviated day of the week
    SEVEN_DAY_TIME=$(date -d @"$SEVEN_DAY_RESET_UNIX" "+%a %I:%M%p")
    SEVEN_DAY_STR="📅 7d: ${SEVEN_DAY_PCT}% (resets @ ${SEVEN_DAY_TIME})"
else
    SEVEN_DAY_STR="📅 7d: ${SEVEN_DAY_PCT}%"
fi

BUFFER_STATUS="${FIVE_HR_STR} | ${SEVEN_DAY_STR}"
# ---------------------------------------

# --- FINAL OUTPUT: MULTI-LINE DASHBOARD ---
echo -e "${ICON} ${COLOR}Context: ${CONTEXT_PCT}%${RESET}  |  🧠 ${MODEL}"
echo -e "${FIVE_HR_STR}  |  ${SEVEN_DAY_STR}  |  ${BUDGET_PREDICTION}"

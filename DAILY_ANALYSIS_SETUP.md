# Daily Analysis Logger Setup Guide

## Overview

The Daily Analysis Logger automatically generates comprehensive written market analysis during opening hours (9:30 AM - 11:30 AM EST) and stores it in a searchable history. The system uses LLMs (OpenAI or Anthropic) to generate professional, detailed analysis, or falls back to rule-based analysis if LLM APIs are not configured.

## Features

✅ **Automated Analysis Generation**
- Runs every 15 minutes during 9:30 AM - 11:30 AM EST
- Collects real-time market data
- Performs price action and ORB analysis
- Generates written analysis using LLMs or rule-based system

✅ **Comprehensive Analysis Includes:**
- Market overview and conditions
- Key price movements and volume patterns
- Technical analysis (support/resistance, patterns)
- Trading opportunities and risk considerations
- Short-term outlook

✅ **Historical Logging**
- Daily logs stored in `data/analysis_logs/`
- JSON format for easy retrieval
- Searchable by date
- Multiple analyses per day (every 15 minutes)

✅ **Dashboard Integration**
- View historical analyses
- Browse by date
- Display formatted analysis with market data summaries

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `openai>=1.0.0` - For OpenAI GPT models
- `anthropic>=0.18.0` - For Anthropic Claude models
- `schedule>=1.2.0` - For automated scheduling

### 2. Configure LLM API Keys (Optional but Recommended)

#### Option A: OpenAI (Recommended for cost-effectiveness)

1. Sign up at https://platform.openai.com/
2. Get your API key
3. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

#### Option B: Anthropic Claude

1. Sign up at https://console.anthropic.com/
2. Get your API key
3. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

#### Option C: No LLM (Rule-Based)

If no API keys are set, the system will use rule-based analysis generation. This is less sophisticated but still provides valuable insights.

### 3. Create Analysis Directory

The system will automatically create `data/analysis_logs/` directory, but you can create it manually:

```bash
mkdir -p data/analysis_logs
```

### 4. Integration with Dashboard

Add to your main dashboard file (`dashbord_blur.py`):

```python
# At the top, import the components
from analysis_dashboard_component import create_analysis_history_component, get_analysis_callback
from analysis_scheduler import start_scheduler

# In create_dash_app(), add the analysis history component to layout
analysis_history = create_analysis_history_component()

# Add callback for updating analysis display
@app.callback(
    Output("analysis-content-container", "children"),
    Input("analysis-date-dropdown", "value")
)
def update_analysis_display(selected_date):
    from analysis_dashboard_component import format_analysis_for_display
    from daily_analysis_logger import get_daily_analyses
    
    if not selected_date:
        return html.Div("No date selected.")
    
    analyses = get_daily_analyses(selected_date)
    return format_analysis_for_display(analyses)

# Start scheduler when app initializes
start_scheduler()
```

### 5. Manual Analysis Generation

You can manually trigger analysis at any time:

```python
from daily_analysis_logger import generate_comprehensive_analysis, save_daily_analysis

# Generate analysis
analysis = generate_comprehensive_analysis(
    symbols=["ES=F", "NQ=F", "GC=F"],
    use_llm=True  # Set to False for rule-based
)

# Save to daily log
save_daily_analysis(analysis)
```

## Usage

### Automated Mode

The scheduler runs automatically in the background:
- Checks every minute for scheduled tasks
- Runs analysis every 15 minutes during 9:30 AM - 11:30 AM EST
- Also runs at market open (9:30 AM EST)

### Manual Mode

Run analysis manually:

```python
from daily_analysis_logger import run_automated_analysis

# Run if during analysis hours
analysis = run_automated_analysis(force=False)

# Force run regardless of time
analysis = run_automated_analysis(force=True)
```

### Viewing Historical Analysis

```python
from daily_analysis_logger import get_daily_analyses, get_all_analysis_dates

# Get all dates with analyses
dates = get_all_analysis_dates()

# Get analyses for a specific date
analyses = get_daily_analyses("2024-11-24")

# Each analysis contains:
# - timestamp
# - date
# - time_est
# - market_data
# - price_action_analysis
# - orb_analysis
# - written_analysis
# - analysis_type
```

## File Structure

```
data/
└── analysis_logs/
    ├── analysis_2024-11-24.json
    ├── analysis_2024-11-25.json
    └── ...
```

Each JSON file contains an array of analysis objects for that day.

## Analysis Format

Each analysis object contains:

```json
{
  "timestamp": "2024-11-24T14:30:00Z",
  "date": "2024-11-24",
  "time_est": "09:30:00",
  "market_data": {
    "symbols": {
      "ES=F": {
        "current_price": 6654.25,
        "open_price": 6640.00,
        "price_change": 14.25,
        "price_change_pct": 0.21,
        ...
      }
    }
  },
  "price_action_analysis": {
    "trend_analysis": {...},
    "support_resistance": {...},
    ...
  },
  "orb_analysis": {
    "orb_high": 6660.00,
    "orb_low": 6635.00,
    "breakout_status": "INSIDE_RANGE",
    ...
  },
  "written_analysis": "Full written analysis text...",
  "analysis_type": "LLM" or "RULE_BASED"
}
```

## Cost Considerations

### OpenAI (GPT-4o-mini)
- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens
- Estimated cost per analysis: ~$0.01-0.02
- Daily cost (2 hours, 8 analyses): ~$0.08-0.16

### Anthropic (Claude Haiku)
- ~$0.25 per 1M input tokens
- ~$1.25 per 1M output tokens
- Estimated cost per analysis: ~$0.02-0.03
- Daily cost (2 hours, 8 analyses): ~$0.16-0.24

### Rule-Based (No Cost)
- Free alternative
- Less sophisticated but still valuable
- Good for testing or cost-conscious deployments

## Troubleshooting

### Analysis not generating
1. Check if during analysis hours (9:30 AM - 11:30 AM EST)
2. Verify API keys are set correctly
3. Check logs for errors
4. Ensure data directory exists and is writable

### LLM errors
1. Verify API key is valid
2. Check API quota/limits
3. Ensure internet connection
4. System will fall back to rule-based if LLM fails

### Scheduler not running
1. Ensure `start_scheduler()` is called in your app
2. Check that schedule library is installed
3. Verify timezone settings

## Best Practices

1. **API Key Security**: Never commit API keys to git. Use environment variables.
2. **Storage**: Regularly backup `data/analysis_logs/` directory
3. **Monitoring**: Set up alerts for analysis generation failures
4. **Cost Management**: Monitor API usage if using LLMs
5. **Testing**: Test with rule-based analysis first before enabling LLMs

## Future Enhancements

Potential improvements:
- Email/SMS alerts for key insights
- Custom analysis templates
- Multi-timeframe analysis
- Sentiment analysis integration
- Performance metrics tracking
- Export to PDF/Excel
- Search functionality across all analyses


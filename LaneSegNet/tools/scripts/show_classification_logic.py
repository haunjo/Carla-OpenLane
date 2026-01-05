"""
Show how scene tokens are classified into Weather and Time categories
"""

# Example tokens from the dataset
example_tokens = [
    ['dusk', 'heavy_rain', 'cloudy', 'downtown'],
    ['afternoon', 'clear', 'suburban'],
    ['night', 'clear', 'urban'],
    ['dusk', 'rain', 'highway'],
    ['afternoon', 'cloudy', 'downtown'],
    ['night', 'heavy_rain', 'cloudy', 'suburban'],
]

print("="*70)
print("CLASSIFICATION LOGIC")
print("="*70)

print("\n1. TIME CLASSIFICATION:")
print("-" * 70)
print("Mapping rules:")
print("  • 'afternoon' or 'noon' → Noon")
print("  • 'dusk' or 'sunset' → Sunset")
print("  • 'night' → Night")

print("\n2. WEATHER CLASSIFICATION (priority order):")
print("-" * 70)
print("Priority 1: 'heavy_rain' → Heavy Rain")
print("Priority 2: 'rain' or 'light_rain' → Rain")
print("Priority 3: 'fog' or 'heavy_fog' → Foggy")
print("Priority 4: 'cloudy' → Cloudy")
print("Priority 5: (default) → Clear")
print()
print("NOTE: If 'heavy_rain' is present, it takes priority over 'cloudy'")

print("\n" + "="*70)
print("CLASSIFICATION EXAMPLES")
print("="*70)

for i, tokens in enumerate(example_tokens, 1):
    print(f"\nExample {i}:")
    print(f"  Tokens: {tokens}")

    # Time classification
    time_result = None
    for token in tokens:
        if token in ['afternoon', 'noon']:
            time_result = 'Noon'
            time_token = token
            break
        elif token in ['dusk', 'sunset']:
            time_result = 'Sunset'
            time_token = token
            break
        elif token == 'night':
            time_result = 'Night'
            time_token = token
            break

    # Weather classification (with priority)
    has_heavy_rain = 'heavy_rain' in tokens
    has_rain = 'rain' in tokens or 'light_rain' in tokens
    has_cloudy = 'cloudy' in tokens
    has_fog = 'fog' in tokens or 'heavy_fog' in tokens

    if has_heavy_rain:
        weather_result = 'Heavy Rain'
        weather_reason = "found 'heavy_rain' (priority 1)"
    elif has_rain:
        weather_result = 'Rain'
        weather_reason = "found 'rain' (priority 2)"
    elif has_fog:
        weather_result = 'Foggy'
        weather_reason = "found 'fog' (priority 3)"
    elif has_cloudy:
        weather_result = 'Cloudy'
        weather_reason = "found 'cloudy' (priority 4)"
    else:
        weather_result = 'Clear'
        weather_reason = "no weather token (default)"

    print(f"  Time: '{time_token}' → {time_result}")
    print(f"  Weather: {weather_result} ({weather_reason})")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("""
Example: ['dusk', 'heavy_rain', 'cloudy', 'downtown']
  → Time: 'dusk' → Sunset
  → Weather: Heavy Rain (NOT Cloudy!)

Even though 'cloudy' is present, 'heavy_rain' has higher priority,
so the scene is classified as "Heavy Rain" not "Cloudy".
This is because heavy rain implies cloudy conditions anyway.
""")

print("\n" + "="*70)
print("ACTUAL DATASET DISTRIBUTION (6,649 samples)")
print("="*70)
print("""
Weather:
  Clear       : 1,880 (28.3%)
  Cloudy      : 2,006 (30.2%)
  Rain        :   800 (12.0%)
  Heavy Rain  : 1,963 (29.5%)

Time-of-Day:
  Noon        : 2,671 (40.2%)
  Sunset      : 2,783 (41.9%)
  Night       : 1,195 (18.0%)

Scene Type:
  Downtown    : 2,677 (40.3%)
  Suburban    : 2,631 (39.6%)
  Urban       :   680 (10.2%)
  Highway     :   661 ( 9.9%)
""")

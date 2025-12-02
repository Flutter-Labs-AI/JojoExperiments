# -*- coding: utf-8 -*-
"""Copy of article_real2.ipynb

"""











import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

print("="*80)
print("MICROWEIGHT SYSTEM WITH TEMPORAL DECAY - CORRECTED")
print("="*80)

# Load relevancy feedback
relevance_df = pd.read_csv('/translated_articles_relevance.csv',
                           header=None,
                           names=['article_id', 'relevant', 'org_id', 'asset_id', 'incident_id'])

# Load article metadata (has timestamps and T2 categories!)
articles_df = pd.read_csv('/found_articles_nov24a.csv')
print(f"✓ Loaded {len(articles_df)} articles with metadata")

print(f"\n✓ Loaded {len(relevance_df)} relevancy marks")
print(f"✓ Loaded {len(articles_df)} articles with timestamps and T2 categories")

# Merge to get timestamps and T2 categories for relevant marks
merged = relevance_df.merge(
    articles_df[['id', 'utc_datetime', 'category', 'code']],
    left_on='article_id',
    right_on='id',
    how='left'
)

# Keep ALL marks (both TRUE and FALSE), just need timestamps
all_marks = merged.dropna(subset=['utc_datetime']).copy()

print(f"✓ Matched {len(all_marks)} marks (both up/down) with timestamps and categories")
print(f"  - TRUE marks (up-votes): {(all_marks['relevant'] == True).sum()}")
print(f"  - FALSE marks (down-votes): {(all_marks['relevant'] == False).sum()}")

if len(all_marks) == 0:
    print("\n⚠️  No marks matched with article timestamps")
    print("Cannot proceed without data")
else:
    # Configuration
    DECAY_RATE = 0.05  # 5% daily decay
    LEARNING_RATE = 0.1  # 10% weight change per click
    BASELINE = 1.0  # Neutral weight
    MIN_WEIGHT = 0.5  # Minimum weight (for down-voted categories)
    MAX_WEIGHT = 2.0  # Maximum weight (for up-voted categories)

    print(f"\nConfiguration:")
    print(f"  Decay rate: {DECAY_RATE} (5% daily)")
    print(f"  Learning rate: {LEARNING_RATE} (10% per mark)")
    print(f"  Baseline: {BASELINE}")
    print(f"  Weight range: [{MIN_WEIGHT}, {MAX_WEIGHT}]")

    # Reference date (use latest article date as "now")
    reference_date = articles_df['utc_datetime'].max()
    print(f"  Reference date (now): {reference_date}")

    # Calculate microweights with temporal decay for each asset
    print("\n" + "="*80)
    print("ASSET-SPECIFIC T2 CATEGORY MICROWEIGHTS WITH TEMPORAL DECAY")
    print("="*80)

    asset_microweights = {}

    for asset_id in sorted(all_marks['asset_id'].unique()):
        asset_marks = all_marks[all_marks['asset_id'] == asset_id].copy()
        asset_marks = asset_marks.sort_values('utc_datetime')

        print(f"\n{'='*80}")
        print(f"ASSET {asset_id}")
        print(f"{'='*80}")
        print(f"Total marks: {len(asset_marks)} (↑{(asset_marks['relevant']==True).sum()}, ↓{(asset_marks['relevant']==False).sum()})")

        # Accumulate decayed contributions per T2 category
        category_contributions = defaultdict(float)

        # Process each mark - each contributes based on its own age
        for idx, mark in asset_marks.iterrows():
            category = mark['category']
            mark_date = mark['utc_datetime']
            is_relevant = mark['relevant']

            # Skip if no category (shouldn't happen but safeguard)
            if pd.isna(category):
                continue

            # Direction: +1 for up-vote (TRUE), -1 for down-vote (FALSE)
            direction = +1 if is_relevant else -1

            # Calculate age of THIS specific mark
            days_old = (reference_date - mark_date).days
            decay_factor = (1 - DECAY_RATE) ** days_old

            # This mark's decayed contribution
            contribution = LEARNING_RATE * direction * decay_factor

            # Accumulate
            category_contributions[category] += contribution

            print(f"\n  Mark from {mark_date.date()} ({days_old} days ago):")
            print(f"    Article: {mark['article_id']}")
            print(f"    Category: {category} ({mark['code']})")
            print(f"    Vote: {'↑ UP' if is_relevant else '↓ DOWN'} (direction: {direction:+.1f})")
            print(f"    Decay factor: {decay_factor:.4f}")
            print(f"    Contribution: {contribution:+.4f}")
            print(f"    Running total for {category}: {category_contributions[category]:+.4f}")

        # Apply baseline and clamp to valid range
        final_weights = {}
        for category, contribution in category_contributions.items():
            weight = BASELINE + contribution
            clamped_weight = np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)
            final_weights[category] = clamped_weight

        print(f"\n  {'─'*76}")
        print(f"  FINAL T2 CATEGORY MICROWEIGHTS (after decay and clamping):")
        print(f"  {'─'*76}")

        # Sort by weight descending
        sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)

        for category, weight in sorted_weights:
            raw_contribution = category_contributions[category]
            impact = "🔥 BOOSTED" if weight > BASELINE else "❄️  SUPPRESSED" if weight < BASELINE else "⚪ NEUTRAL"
            print(f"    {category:20s}: {weight:.4f} (raw: {BASELINE + raw_contribution:+.4f}) {impact}")

        # Store for this asset
        asset_microweights[asset_id] = {
            'weights': final_weights,
            'contributions': dict(category_contributions),
            'last_mark_date': asset_marks['utc_datetime'].max(),
            'total_marks': len(asset_marks),
            'up_votes': (asset_marks['relevant'] == True).sum(),
            'down_votes': (asset_marks['relevant'] == False).sum()
        }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: ASSET MICROWEIGHTS")
    print("="*80)

    for asset_id, data in asset_microweights.items():
        weights = data['weights']

        print(f"\nAsset {asset_id}:")
        print(f"  Activity: {data['total_marks']} marks (↑{data['up_votes']}, ↓{data['down_votes']})")
        print(f"  Last mark: {data['last_mark_date'].date()} ({(reference_date - data['last_mark_date']).days} days ago)")
        print(f"  Affected categories: {len(weights)}")

        # Show top boosted and suppressed
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        if sorted_weights:
            print(f"  Most boosted:")
            for cat, weight in sorted_weights[:3]:
                if weight > BASELINE:
                    print(f"    {cat}: {weight:.4f}")

            print(f"  Most suppressed:")
            suppressed = [(cat, w) for cat, w in sorted_weights if w < BASELINE]
            for cat, weight in suppressed[-3:]:
                print(f"    {cat}: {weight:.4f}")

    # Export to JSON for production use
    print("\n" + "="*80)
    print("EXPORTING FOR PRODUCTION")
    print("="*80)

    # Create production-ready format
    production_weights = {}
    for asset_id, data in asset_microweights.items():
        production_weights[int(asset_id)] = {
            'microweights': {cat: float(weight) for cat, weight in data['weights'].items()},
            'metadata': {
                'last_updated': data['last_mark_date'].isoformat(),
                'total_marks': int(data['total_marks']),
                'up_votes': int(data['up_votes']),
                'down_votes': int(data['down_votes'])
            }
        }

    import json
    output_path = '/asset_microweights.json'
    with open(output_path, 'w') as f:
        json.dump(production_weights, f, indent=2)

    print(f"✓ Exported microweights to: {output_path}")
    print(f"✓ {len(production_weights)} assets with custom microweights")

    print("\n" + "="*80)
    print("✓ COMPLETE - READY FOR SCORING INTEGRATION")
    print("="*80)
    print("\nHow to use in scoring:")
    print("  1. Load asset_microweights.json")
    print("  2. For each article matched to asset:")
    print("     article_t2_category = article['category']")
    print("     microweight = asset_weights[asset_id].get(article_t2_category, 1.0)")
    print("     adjusted_score = base_score * microweight")
    print("  3. Categories with weight > 1.0 boost score (user wants more)")
    print("  4. Categories with weight < 1.0 suppress score (user wants less)")





import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

print("="*80)
print("MICROWEIGHT SYSTEM WITH TEMPORAL DECAY - CORRECTED")
print("="*80)

# Load relevancy feedback
relevance_df = pd.read_csv('/content/drive/MyDrive/1_DataSc_Funnel/events/articles_relevancy/translated_articles_relevance.csv',
                           header=None,
                           names=['article_id', 'relevant', 'org_id', 'asset_id', 'incident_id'])

# Load article metadata (has timestamps and T2 categories!)
articles_df = pd.read_csv('/content/drive/MyDrive/1_DataSc_Funnel/events/articles_relevancy/found_articles_nov24a.csv')
print(f"✓ Loaded {len(articles_df)} articles with metadata")


# Convert datetime - handle potential parsing issues
try:
    articles_df['utc_datetime'] = pd.to_datetime(articles_df['utc_datetime'], errors='coerce')
except:
    print("⚠️  Warning: Could not parse some datetime values")

print(f"\n✓ Loaded {len(relevance_df)} relevancy marks")
print(f"✓ Loaded {len(articles_df)} articles with timestamps and T2 categories")

# Merge to get timestamps and T2 categories for relevant marks
merged = relevance_df.merge(
    articles_df[['id', 'utc_datetime', 'category', 'code']],
    left_on='article_id',
    right_on='id',
    how='left'
)

# Ensure utc_datetime is datetime type after merge
merged['utc_datetime'] = pd.to_datetime(merged['utc_datetime'], errors='coerce')

# Keep ALL marks (both TRUE and FALSE), just need timestamps
all_marks = merged.dropna(subset=['utc_datetime']).copy()

print(f"✓ Matched {len(all_marks)} marks (both up/down) with timestamps and categories")
print(f"  - TRUE marks (up-votes): {(all_marks['relevant'] == True).sum()}")
print(f"  - FALSE marks (down-votes): {(all_marks['relevant'] == False).sum()}")

if len(all_marks) == 0:
    print("\n⚠️  No marks matched with article timestamps")
    print("Cannot proceed without data")
else:
    # Configuration
    DECAY_RATE = 0.05  # 5% daily decay
    LEARNING_RATE = 0.1  # 10% weight change per click
    BASELINE = 1.0  # Neutral weight
    MIN_WEIGHT = 0.5  # Minimum weight (for down-voted categories)
    MAX_WEIGHT = 2.0  # Maximum weight (for up-voted categories)

    print(f"\nConfiguration:")
    print(f"  Decay rate: {DECAY_RATE} (5% daily)")
    print(f"  Learning rate: {LEARNING_RATE} (10% per mark)")
    print(f"  Baseline: {BASELINE}")
    print(f"  Weight range: [{MIN_WEIGHT}, {MAX_WEIGHT}]")

    # Reference date (use latest article date as "now")
    reference_date = pd.to_datetime(articles_df['utc_datetime']).max()
    if pd.isna(reference_date):
        # Fallback to current date if no valid dates
        reference_date = pd.Timestamp.now()
    print(f"  Reference date (now): {reference_date}")

    # Calculate microweights with temporal decay for each asset
    print("\n" + "="*80)
    print("ASSET-SPECIFIC T2 CATEGORY MICROWEIGHTS WITH TEMPORAL DECAY")
    print("="*80)

    asset_microweights = {}

    for asset_id in sorted(all_marks['asset_id'].unique()):
        asset_marks = all_marks[all_marks['asset_id'] == asset_id].copy()
        asset_marks = asset_marks.sort_values('utc_datetime')

        print(f"\n{'='*80}")
        print(f"ASSET {asset_id}")
        print(f"{'='*80}")
        print(f"Total marks: {len(asset_marks)} (↑{(asset_marks['relevant']==True).sum()}, ↓{(asset_marks['relevant']==False).sum()})")

        # Accumulate decayed contributions per T2 category
        category_contributions = defaultdict(float)

        # Process each mark - each contributes based on its own age
        for idx, mark in asset_marks.iterrows():
            category = mark['category']
            mark_date = mark['utc_datetime']
            is_relevant = mark['relevant']

            # Skip if no category (shouldn't happen but safeguard)
            if pd.isna(category):
                continue

            # Direction: +1 for up-vote (TRUE), -1 for down-vote (FALSE)
            direction = +1 if is_relevant else -1

            # Calculate age of THIS specific mark
            days_old = (reference_date - mark_date).days
            decay_factor = (1 - DECAY_RATE) ** days_old

            # This mark's decayed contribution
            contribution = LEARNING_RATE * direction * decay_factor

            # Accumulate
            category_contributions[category] += contribution

            print(f"\n  Mark from {mark_date.date()} ({days_old} days ago):")
            print(f"    Article: {mark['article_id']}")
            print(f"    Category: {category} ({mark['code']})")
            print(f"    Vote: {'↑ UP' if is_relevant else '↓ DOWN'} (direction: {direction:+.1f})")
            print(f"    Decay factor: {decay_factor:.4f}")
            print(f"    Contribution: {contribution:+.4f}")
            print(f"    Running total for {category}: {category_contributions[category]:+.4f}")

        # Apply baseline and clamp to valid range
        final_weights = {}
        for category, contribution in category_contributions.items():
            weight = BASELINE + contribution
            clamped_weight = np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)
            final_weights[category] = clamped_weight

        print(f"\n  {'─'*76}")
        print(f"  FINAL T2 CATEGORY MICROWEIGHTS (after decay and clamping):")
        print(f"  {'─'*76}")

        # Sort by weight descending
        sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)

        for category, weight in sorted_weights:
            raw_contribution = category_contributions[category]
            impact = "🔥 BOOSTED" if weight > BASELINE else "❄️  SUPPRESSED" if weight < BASELINE else "⚪ NEUTRAL"
            print(f"    {category:20s}: {weight:.4f} (raw: {BASELINE + raw_contribution:+.4f}) {impact}")

        # Store for this asset
        asset_microweights[asset_id] = {
            'weights': final_weights,
            'contributions': dict(category_contributions),
            'last_mark_date': asset_marks['utc_datetime'].max(),
            'total_marks': len(asset_marks),
            'up_votes': (asset_marks['relevant'] == True).sum(),
            'down_votes': (asset_marks['relevant'] == False).sum()
        }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: ASSET MICROWEIGHTS")
    print("="*80)

    for asset_id, data in asset_microweights.items():
        weights = data['weights']

        print(f"\nAsset {asset_id}:")
        print(f"  Activity: {data['total_marks']} marks (↑{data['up_votes']}, ↓{data['down_votes']})")
        print(f"  Last mark: {data['last_mark_date'].date()} ({(reference_date - data['last_mark_date']).days} days ago)")
        print(f"  Affected categories: {len(weights)}")

        # Show top boosted and suppressed
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        if sorted_weights:
            print(f"  Most boosted:")
            for cat, weight in sorted_weights[:3]:
                if weight > BASELINE:
                    print(f"    {cat}: {weight:.4f}")

            print(f"  Most suppressed:")
            suppressed = [(cat, w) for cat, w in sorted_weights if w < BASELINE]
            for cat, weight in suppressed[-3:]:
                print(f"    {cat}: {weight:.4f}")

    # Export to JSON for production use
    print("\n" + "="*80)
    print("EXPORTING FOR PRODUCTION")
    print("="*80)

    # Create production-ready format
    production_weights = {}
    for asset_id, data in asset_microweights.items():
        production_weights[int(asset_id)] = {
            'microweights': {cat: float(weight) for cat, weight in data['weights'].items()},
            'metadata': {
                'last_updated': data['last_mark_date'].isoformat(),
                'total_marks': int(data['total_marks']),
                'up_votes': int(data['up_votes']),
                'down_votes': int(data['down_votes'])
            }
        }

    import json
    output_path = '/content/drive/MyDrive/1_DataSc_Funnel/events/articles_relevancy_realtime/asset_microweights.json'
    with open(output_path, 'w') as f:
        json.dump(production_weights, f, indent=2)

    print(f"✓ Exported microweights to: {output_path}")
    print(f"✓ {len(production_weights)} assets with custom microweights")

    print("\n" + "="*80)
    print("✓ COMPLETE - READY FOR SCORING INTEGRATION")
    print("="*80)
    print("\nHow to use in scoring:")
    print("  1. Load asset_microweights.json")
    print("  2. For each article matched to asset:")
    print("     article_t2_category = article['category']")
    print("     microweight = asset_weights[asset_id].get(article_t2_category, 1.0)")
    print("     adjusted_score = base_score * microweight")
    print("  3. Categories with weight > 1.0 boost score (user wants more)")
    print("  4. Categories with weight < 1.0 suppress score (user wants less)")



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

print("="*80)
print("MICROWEIGHT SYSTEM WITH TEMPORAL DECAY - CORRECTED")
print("="*80)

# Load relevancy feedback
relevance_df = pd.read_csv('/content/drive/MyDrive/1_DataSc_Funnel/events/articles_relevancy/translated_articles_relevance.csv',
                           header=None,
                           names=['article_id', 'relevant', 'org_id', 'asset_id', 'incident_id'])

# Load article metadata (has timestamps and T2 categories!)
articles_df = pd.read_csv('/content/drive/MyDrive/1_DataSc_Funnel/events/articles_relevancy/found_articles_nov24a.csv')
print(f"✓ Loaded {len(articles_df)} articles with metadata")


# Convert datetime - handle potential parsing issues
try:
    articles_df['utc_datetime'] = pd.to_datetime(articles_df['utc_datetime'], errors='coerce')
except:
    print("⚠️  Warning: Could not parse some datetime values")

print(f"\n✓ Loaded {len(relevance_df)} relevancy marks")
print(f"✓ Loaded {len(articles_df)} articles with timestamps and T2 categories")

# Merge to get timestamps and T2 categories for relevant marks
merged = relevance_df.merge(
    articles_df[['id', 'utc_datetime', 'category', 'code']],
    left_on='article_id',
    right_on='id',
    how='left'
)

# Ensure utc_datetime is datetime type after merge
merged['utc_datetime'] = pd.to_datetime(merged['utc_datetime'], errors='coerce')

# Keep ALL marks (both TRUE and FALSE), just need timestamps
all_marks = merged.dropna(subset=['utc_datetime']).copy()

print(f"✓ Matched {len(all_marks)} marks (both up/down) with timestamps and categories")
print(f"  - TRUE marks (up-votes): {(all_marks['relevant'] == True).sum()}")
print(f"  - FALSE marks (down-votes): {(all_marks['relevant'] == False).sum()}")

if len(all_marks) == 0:
    print("\n⚠️  No marks matched with article timestamps")
    print("Cannot proceed without data")
else:
    # Configuration
    DECAY_RATE = 0.05  # 5% daily decay
    LEARNING_RATE = 0.1  # 10% weight change per click
    BASELINE = 1.0  # Neutral weight
    MIN_WEIGHT = 0.5  # Minimum weight (for down-voted categories)
    MAX_WEIGHT = 2.0  # Maximum weight (for up-voted categories)

    print(f"\nConfiguration:")
    print(f"  Decay rate: {DECAY_RATE} (5% daily)")
    print(f"  Learning rate: {LEARNING_RATE} (10% per mark)")
    print(f"  Baseline: {BASELINE}")
    print(f"  Weight range: [{MIN_WEIGHT}, {MAX_WEIGHT}]")

    # Reference date (use latest article date as "now")
    reference_date = pd.to_datetime(articles_df['utc_datetime']).max()
    if pd.isna(reference_date):
        # Fallback to current date if no valid dates
        reference_date = pd.Timestamp.now()
    print(f"  Reference date (now): {reference_date}")

    # Calculate microweights with temporal decay for each asset
    print("\n" + "="*80)
    print("ASSET-SPECIFIC T2 CATEGORY MICROWEIGHTS WITH TEMPORAL DECAY")
    print("="*80)

    asset_microweights = {}

    for asset_id in sorted(all_marks['asset_id'].unique()):
        asset_marks = all_marks[all_marks['asset_id'] == asset_id].copy()
        asset_marks = asset_marks.sort_values('utc_datetime')

        print(f"\n{'='*80}")
        print(f"ASSET {asset_id}")
        print(f"{'='*80}")
        print(f"Total marks: {len(asset_marks)} (↑{(asset_marks['relevant']==True).sum()}, ↓{(asset_marks['relevant']==False).sum()})")

        # Accumulate decayed contributions per T2 category
        category_contributions = defaultdict(float)

        # Process each mark - each contributes based on its own age
        for idx, mark in asset_marks.iterrows():
            category = mark['category']
            mark_date = mark['utc_datetime']
            is_relevant = mark['relevant']

            # Skip if no category (shouldn't happen but safeguard)
            if pd.isna(category):
                continue

            # Direction: +1 for up-vote (TRUE), -1 for down-vote (FALSE)
            direction = +1 if is_relevant else -1

            # Calculate age of THIS specific mark
            days_old = (reference_date - mark_date).days
            decay_factor = (1 - DECAY_RATE) ** days_old

            # This mark's decayed contribution
            contribution = LEARNING_RATE * direction * decay_factor

            # Accumulate
            category_contributions[category] += contribution

            print(f"\n  Mark from {mark_date.date()} ({days_old} days ago):")
            print(f"    Article: {mark['article_id']}")
            print(f"    Category: {category} ({mark['code']})")
            print(f"    Vote: {'↑ UP' if is_relevant else '↓ DOWN'} (direction: {direction:+.1f})")
            print(f"    Decay factor: {decay_factor:.4f}")
            print(f"    Contribution: {contribution:+.4f}")
            print(f"    Running total for {category}: {category_contributions[category]:+.4f}")

        # Apply baseline and clamp to valid range
        final_weights = {}
        for category, contribution in category_contributions.items():
            weight = BASELINE + contribution
            clamped_weight = np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)
            final_weights[category] = clamped_weight

        print(f"\n  {'─'*76}")
        print(f"  FINAL T2 CATEGORY MICROWEIGHTS (after decay and clamping):")
        print(f"  {'─'*76}")

        # Sort by weight descending
        sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)

        for category, weight in sorted_weights:
            raw_contribution = category_contributions[category]
            impact = "🔥 BOOSTED" if weight > BASELINE else "❄️  SUPPRESSED" if weight < BASELINE else "⚪ NEUTRAL"
            print(f"    {category:20s}: {weight:.4f} (raw: {BASELINE + raw_contribution:+.4f}) {impact}")

        # Store for this asset
        asset_microweights[asset_id] = {
            'weights': final_weights,
            'contributions': dict(category_contributions),
            'last_mark_date': asset_marks['utc_datetime'].max(),
            'total_marks': len(asset_marks),
            'up_votes': (asset_marks['relevant'] == True).sum(),
            'down_votes': (asset_marks['relevant'] == False).sum()
        }

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: ASSET MICROWEIGHTS")
    print("="*80)

    for asset_id, data in asset_microweights.items():
        weights = data['weights']

        print(f"\nAsset {asset_id}:")
        print(f"  Activity: {data['total_marks']} marks (↑{data['up_votes']}, ↓{data['down_votes']})")
        print(f"  Last mark: {data['last_mark_date'].date()} ({(reference_date - data['last_mark_date']).days} days ago)")
        print(f"  Affected categories: {len(weights)}")

        # Show top boosted and suppressed
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        if sorted_weights:
            print(f"  Most boosted:")
            for cat, weight in sorted_weights[:3]:
                if weight > BASELINE:
                    print(f"    {cat}: {weight:.4f}")

            print(f"  Most suppressed:")
            suppressed = [(cat, w) for cat, w in sorted_weights if w < BASELINE]
            for cat, weight in suppressed[-3:]:
                print(f"    {cat}: {weight:.4f}")

    # Export to JSON for production use
    print("\n" + "="*80)
    print("EXPORTING FOR PRODUCTION")
    print("="*80)

    # Create production-ready format
    production_weights = {}
    for asset_id, data in asset_microweights.items():
        production_weights[int(asset_id)] = {
            'microweights': {cat: float(weight) for cat, weight in data['weights'].items()},
            'metadata': {
                'last_updated': data['last_mark_date'].isoformat(),
                'total_marks': int(data['total_marks']),
                'up_votes': int(data['up_votes']),
                'down_votes': int(data['down_votes'])
            }
        }

    import json
    output_path = '/content/drive/MyDrive/1_DataSc_Funnel/events/articles_relevancy_realtime/asset_microweights.json'
    with open(output_path, 'w') as f:
        json.dump(production_weights, f, indent=2)

    print(f"✓ Exported microweights to: {output_path}")
    print(f"✓ {len(production_weights)} assets with custom microweights")

    print("\n" + "="*80)
    print("✓ COMPLETE - READY FOR SCORING INTEGRATION")
    print("="*80)
    print("\nHow to use in scoring:")
    print("  1. Load asset_microweights.json")
    print("  2. For each article matched to asset:")
    print("     article_t2_category = article['category']")
    print("     microweight = asset_weights[asset_id].get(article_t2_category, 1.0)")
    print("     adjusted_score = base_score * microweight")
    print("  3. Categories with weight > 1.0 boost score (user wants more)")
    print("  4. Categories with weight < 1.0 suppress score (user wants less)")




import pandas as pd
from datetime import datetime, timedelta
import pytz
import json
from scipy import stats
import numpy as np

def load_and_prepare_data():
    print("Loading sleep data...")
    sleep_df = pd.read_csv('sleep_data_with_duration.csv')
    
    # Convert sleep/wake times to datetime
    sleep_df['date'] = pd.to_datetime(sleep_df['date'])
    
    def convert_time(row, time_col):
        time_str = row[time_col]
        base_date = row['date']
        
        # Convert time string to hours and minutes
        hours, minutes = map(int, time_str.split(':'))
        
        # For sleep time: if it's AM (0-11), it's the next day
        if time_col == 'sleep-time' and hours < 12:
            date = base_date + timedelta(days=1)
        # For wake time: if it's AM but sleep was PM, it's the next day
        elif time_col == 'wake-time':
            sleep_hours = int(row['sleep-time'].split(':')[0])
            if hours < 12 and sleep_hours >= 12:
                date = base_date + timedelta(days=1)
            else:
                date = base_date
        else:
            date = base_date
            
        # Create timezone-aware timestamp in UTC
        timestamp = pd.Timestamp.combine(date, pd.Timestamp(f"{hours:02d}:{minutes:02d}").time())
        return timestamp.tz_localize('UTC')
    
    print("Converting sleep times to datetime...")
    sleep_df['sleep_datetime'] = sleep_df.apply(lambda row: convert_time(row, 'sleep-time'), axis=1)
    sleep_df['wake_datetime'] = sleep_df.apply(lambda row: convert_time(row, 'wake-time'), axis=1)

    print("Loading activity data from JSON files...")
    search_activities = []
    watch_activities = []
    chrome_activities = []
    activities = []
    
    # Load search history
    try:
        with open('search-history.json', 'r', encoding='utf-8') as f:
            search_data = json.load(f)
        print(f"Loaded {len(search_data)} search activities")
        
        for item in search_data:
            try:
                # Parse timestamp and ensure UTC timezone
                timestamp = pd.to_datetime(item['time'])
                if timestamp.tz is None:
                    timestamp = timestamp.tz_localize('UTC')
                else:
                    timestamp = timestamp.tz_convert('UTC')
                search_activities.append(timestamp)
                activities.append(timestamp)
            except Exception as e:
                print(f"Error parsing search timestamp {item.get('time', 'unknown')}: {e}")
                continue
    except Exception as e:
        print(f"Error loading search history data: {e}")

    # Load watch history
    try:
        with open('watch-history.json', 'r', encoding='utf-8') as f:
            watch_data = json.load(f)
        print(f"Loaded {len(watch_data)} watch activities")
        
        for item in watch_data:
            try:
                # Parse timestamp and ensure UTC timezone
                timestamp = pd.to_datetime(item['time'])
                if timestamp.tz is None:
                    timestamp = timestamp.tz_localize('UTC')
                else:
                    timestamp = timestamp.tz_convert('UTC')
                watch_activities.append(timestamp)
                activities.append(timestamp)
            except Exception as e:
                print(f"Error parsing watch timestamp {item.get('time', 'unknown')}: {e}")
                continue
    except Exception as e:
        print(f"Error loading watch history data: {e}")
    
    # Load Chrome history
    try:
        with open('Chrome-History.json', 'r', encoding='utf-8') as f:
            chrome_data = json.load(f)
            history_entries = chrome_data.get('Browser History', [])
        print(f"Loaded {len(history_entries)} Chrome history entries")
        
        for entry in history_entries:
            try:
                if isinstance(entry, dict) and 'time_usec' in entry:
                    # Chrome timestamps are always naive, so we can safely localize
                    timestamp = pd.to_datetime(int(entry['time_usec']), unit='us').tz_localize('UTC')
                    chrome_activities.append(timestamp)
                    activities.append(timestamp)
            except Exception as e:
                print(f"Error parsing Chrome timestamp from {entry}: {e}")
                continue
    except Exception as e:
        print(f"Error loading Chrome history data: {e}")

    if not activities:
        print("No activity timestamps found in any file.")
        return None, None, None, None, None
            
    # Create separate DataFrames for each source
    search_df = pd.DataFrame({'Activity Timestamp': search_activities}).sort_values('Activity Timestamp')
    watch_df = pd.DataFrame({'Activity Timestamp': watch_activities}).sort_values('Activity Timestamp')
    chrome_df = pd.DataFrame({'Activity Timestamp': chrome_activities}).sort_values('Activity Timestamp')
    combined_df = pd.DataFrame({'Activity Timestamp': activities}).sort_values('Activity Timestamp')
    
    print(f"\nActivity Data Statistics:")
    print(f"Total activities: {len(combined_df)}")
    if len(combined_df) > 0:
        print(f"Date range: {combined_df['Activity Timestamp'].min()} to {combined_df['Activity Timestamp'].max()}")
        # Print distribution by year
        years = combined_df['Activity Timestamp'].dt.year.value_counts().sort_index()
        print("\nActivities by year:")
        for year, count in years.items():
            print(f"{year}: {count} activities")
    
    return sleep_df, combined_df, search_df, watch_df, chrome_df

def detect_sleep_wake_times(activity_df, date):
    if len(activity_df) == 0:
        print("\nNo activity data found to analyze")
        return None, None
        
    # Get activities for this date +/- 12 hours to reduce processing window
    # Shift the window forward by 12 hours to better align with sleep times
    start_time = date + timedelta(hours=12)  # Changed from -12 to +12
    end_time = date + timedelta(hours=36)    # Keep this the same
    
    try:
        day_activities = activity_df[
            (activity_df['Activity Timestamp'] >= start_time) &
            (activity_df['Activity Timestamp'] <= end_time)
        ]
    except Exception as e:
        print(f"\nError filtering activities: {e}")
        print(f"Activity DataFrame columns: {activity_df.columns}")
        print(f"First few rows: {activity_df.head()}")
        return None, None
    
    if len(day_activities) < 2:
        return None, None
    
    # Group activities into clusters with max 30 min gaps
    day_activities = day_activities.sort_values('Activity Timestamp')
    timestamps = day_activities['Activity Timestamp'].values
    gaps = pd.Series(timestamps[1:]) - pd.Series(timestamps[:-1])
    
    # Find gaps of at least 2 hours
    sleep_gaps = gaps[gaps >= pd.Timedelta(hours=2)]
    
    if len(sleep_gaps) == 0:
        return None, None
    
    # Find gaps that occur during typical sleep periods
    sleep_gaps_with_idx = [(idx, gap) for idx, gap in sleep_gaps.items()]
    valid_sleep_times = []
    
    for gap_idx, gap_duration in sleep_gaps_with_idx:
        # Convert numpy.datetime64 to pandas Timestamp and ensure UTC timezone
        sleep_time = pd.Timestamp(timestamps[gap_idx]).tz_localize('UTC')
        wake_time = pd.Timestamp(timestamps[gap_idx + 1]).tz_localize('UTC')
        sleep_hour = sleep_time.hour
        wake_hour = wake_time.hour
        gap_hours = gap_duration.total_seconds() / 3600
        
        # Sleep time between 8PM and 4AM, wake time between 4AM and noon
        if (((20 <= sleep_hour <= 23) or (0 <= sleep_hour <= 4)) and
            (4 <= wake_hour <= 12) and
            gap_hours >= 3):  # At least 3 hours of sleep
            valid_sleep_times.append((gap_idx, gap_duration))
    
    if not valid_sleep_times:
        return None, None
    
    # Choose the longest valid gap
    longest_gap_idx = max(valid_sleep_times, key=lambda x: x[1])[0]
    
    # Ensure returned timestamps are timezone-aware
    detected_sleep = pd.Timestamp(timestamps[longest_gap_idx]).tz_localize('UTC')
    detected_wake = pd.Timestamp(timestamps[longest_gap_idx + 1]).tz_localize('UTC')
    
    # Adjust dates if needed to match the expected day
    target_date = date.date()
    if detected_sleep.date() < target_date:
        detected_sleep = detected_sleep + timedelta(days=1)
        detected_wake = detected_wake + timedelta(days=1)
    
    return detected_sleep, detected_wake

def analyze_source_accuracy(sleep_df, activity_df, source_name):
    """Analyze accuracy for a specific activity source with detailed statistics"""
    results = []
    total_days = len(sleep_df)
    days_with_detection = 0
    
    sleep_diffs = []
    wake_diffs = []
    sleep_time_errors = []  # For time-of-day analysis
    wake_time_errors = []
    
    for _, sleep_row in sleep_df.iterrows():
        true_sleep = sleep_row['sleep_datetime']
        true_wake = sleep_row['wake_datetime']
        
        detected_sleep, detected_wake = detect_sleep_wake_times(
            activity_df, 
            sleep_row['date'].tz_localize('UTC')
        )
        
        if detected_sleep is not None and detected_wake is not None:
            days_with_detection += 1
            sleep_diff = abs((true_sleep - detected_sleep).total_seconds() / 60)
            wake_diff = abs((true_wake - detected_wake).total_seconds() / 60)
            
            # Calculate time-of-day errors (in minutes from midnight)
            sleep_time_error = (true_sleep.hour * 60 + true_sleep.minute) - (detected_sleep.hour * 60 + detected_sleep.minute)
            wake_time_error = (true_wake.hour * 60 + true_wake.minute) - (detected_wake.hour * 60 + detected_wake.minute)
            
            sleep_diffs.append(sleep_diff)
            wake_diffs.append(wake_diff)
            sleep_time_errors.append(sleep_time_error)
            wake_time_errors.append(wake_time_error)
            
            results.append({
                'date': sleep_row['date'].date(),
                'source': source_name,
                'true_sleep': true_sleep,
                'true_wake': true_wake,
                'detected_sleep': detected_sleep,
                'detected_wake': detected_wake,
                'sleep_diff_minutes': sleep_diff,
                'wake_diff_minutes': wake_diff,
                'sleep_time_error': sleep_time_error,
                'wake_time_error': wake_time_error
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate detailed statistics
    stats_dict = {
        'source': source_name,
        'detection_rate': days_with_detection/total_days,
        'days_detected': days_with_detection,
        'total_days': total_days
    }
    
    if len(results_df) > 0:
        # Basic statistics
        stats_dict.update({
            'sleep_mean_error': np.mean(sleep_diffs),
            'sleep_median_error': np.median(sleep_diffs),
            'sleep_std_error': np.std(sleep_diffs),
            'wake_mean_error': np.mean(wake_diffs),
            'wake_median_error': np.median(wake_diffs),
            'wake_std_error': np.std(wake_diffs),
            'sleep_within_30min': np.mean(np.array(sleep_diffs) <= 30),
            'wake_within_30min': np.mean(np.array(wake_diffs) <= 30),
            
            # Time-of-day accuracy
            'sleep_time_bias': np.mean(sleep_time_errors),  # Positive means detected too late
            'wake_time_bias': np.mean(wake_time_errors),
            
            # Correlation coefficients
            'sleep_time_correlation': stats.pearsonr(
                results_df['true_sleep'].apply(lambda x: x.hour * 60 + x.minute),
                results_df['detected_sleep'].apply(lambda x: x.hour * 60 + x.minute)
            )[0],
            'wake_time_correlation': stats.pearsonr(
                results_df['true_wake'].apply(lambda x: x.hour * 60 + x.minute),
                results_df['detected_wake'].apply(lambda x: x.hour * 60 + x.minute)
            )[0]
        })
    
    return results_df, stats_dict

def smart_combined_detection(sleep_df, search_df, watch_df, chrome_df):
    """Combine data sources intelligently based on their strengths"""
    results = []
    total_days = len(sleep_df)
    days_with_detection = 0
    
    # Different weights for sleep and wake times based on accuracy metrics
    sleep_weights = {
        'Search': 0.5,    # Best sleep accuracy (111.0 min error)
        'Watch': 0.25,    # Worst sleep accuracy (197.8 min error)
        'Chrome': 0.25    # Good time-of-day correlation (0.398)
    }
    
    wake_weights = {
        'Search': 0.2,    # Worst wake accuracy (745.8 min error)
        'Watch': 0.4,     # Best wake accuracy (648.3 min error)
        'Chrome': 0.4     # Best within-30-min accuracy for wake (11.1%)
    }
    
    for _, sleep_row in sleep_df.iterrows():
        true_sleep = sleep_row['sleep_datetime']
        true_wake = sleep_row['wake_datetime']
        date = sleep_row['date'].tz_localize('UTC')
        
        # Get predictions from each source
        predictions = {}
        for source_name, df in [
            ('Search', search_df),
            ('Watch', watch_df),
            ('Chrome', chrome_df)
        ]:
            sleep, wake = detect_sleep_wake_times(df, date)
            if sleep is not None and wake is not None:
                predictions[source_name] = (sleep, wake)
        
        if not predictions:
            continue
        
        # Calculate weighted average times using timestamp values
        total_sleep_weight = 0
        total_wake_weight = 0
        weighted_sleep_ns = 0
        weighted_wake_ns = 0
        
        for source, (sleep, wake) in predictions.items():
            sleep_weight = sleep_weights[source]
            wake_weight = wake_weights[source]
            
            total_sleep_weight += sleep_weight
            total_wake_weight += wake_weight
            weighted_sleep_ns += sleep.value * sleep_weight
            weighted_wake_ns += wake.value * wake_weight
        
        if total_sleep_weight > 0 and total_wake_weight > 0:
            days_with_detection += 1
            # Convert weighted nanoseconds back to timestamps
            detected_sleep = pd.Timestamp(weighted_sleep_ns / total_sleep_weight).tz_localize('UTC')
            detected_wake = pd.Timestamp(weighted_wake_ns / total_wake_weight).tz_localize('UTC')
            
            sleep_diff = abs((true_sleep - detected_sleep).total_seconds() / 60)
            wake_diff = abs((true_wake - detected_wake).total_seconds() / 60)
            
            # Calculate time-of-day errors for consistency with other analysis
            sleep_time_error = (true_sleep.hour * 60 + true_sleep.minute) - (detected_sleep.hour * 60 + detected_sleep.minute)
            wake_time_error = (true_wake.hour * 60 + true_wake.minute) - (detected_wake.hour * 60 + detected_wake.minute)
            
            results.append({
                'date': sleep_row['date'].date(),
                'source': 'Smart Combined',
                'true_sleep': true_sleep,
                'true_wake': true_wake,
                'detected_sleep': detected_sleep,
                'detected_wake': detected_wake,
                'sleep_diff_minutes': sleep_diff,
                'wake_diff_minutes': wake_diff,
                'sleep_time_error': sleep_time_error,
                'wake_time_error': wake_time_error
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate statistics for smart combined approach
    stats_dict = {
        'source': 'Smart Combined',
        'detection_rate': days_with_detection/total_days,
        'days_detected': days_with_detection,
        'total_days': total_days
    }
    
    if len(results_df) > 0:
        stats_dict.update({
            'sleep_mean_error': results_df['sleep_diff_minutes'].mean(),
            'sleep_median_error': results_df['sleep_diff_minutes'].median(),
            'sleep_std_error': results_df['sleep_diff_minutes'].std(),
            'wake_mean_error': results_df['wake_diff_minutes'].mean(),
            'wake_median_error': results_df['wake_diff_minutes'].median(),
            'wake_std_error': results_df['wake_diff_minutes'].std(),
            'sleep_within_30min': (results_df['sleep_diff_minutes'] <= 30).mean(),
            'wake_within_30min': (results_df['wake_diff_minutes'] <= 30).mean(),
            'sleep_time_bias': results_df['sleep_time_error'].mean(),
            'wake_time_bias': results_df['wake_time_error'].mean(),
            'sleep_time_correlation': stats.pearsonr(
                results_df['true_sleep'].apply(lambda x: x.hour * 60 + x.minute),
                results_df['detected_sleep'].apply(lambda x: x.hour * 60 + x.minute)
            )[0],
            'wake_time_correlation': stats.pearsonr(
                results_df['true_wake'].apply(lambda x: x.hour * 60 + x.minute),
                results_df['detected_wake'].apply(lambda x: x.hour * 60 + x.minute)
            )[0]
        })
    
    return results_df, stats_dict

def visualize_results(all_results_df):
    """Create visualizations of the sleep detection results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style - use default style instead of seaborn
    plt.style.use('default')
    
    # 1. Accuracy by source
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=all_results_df, x='source', y='sleep_diff_minutes')
    plt.title('Sleep Time Detection Accuracy by Source')
    plt.ylabel('Difference in Minutes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sleep_accuracy_by_source.png')
    plt.close()

    # 2. Time series of accuracy
    plt.figure(figsize=(15, 6))
    for source in all_results_df['source'].unique():
        source_data = all_results_df[all_results_df['source'] == source]
        plt.plot(source_data['date'], source_data['sleep_diff_minutes'], 
                label=source, marker='o')
    plt.title('Sleep Detection Accuracy Over Time')
    plt.xlabel('Date')
    plt.ylabel('Difference in Minutes')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sleep_accuracy_timeline.png')
    plt.close()

    # 3. Correlation between true and detected times
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=all_results_df, x='true_sleep', y='detected_sleep', 
                    hue='source', alpha=0.6)
    plt.title('True vs Detected Sleep Times')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=all_results_df, x='true_wake', y='detected_wake', 
                    hue='source', alpha=0.6)
    plt.title('True vs Detected Wake Times')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sleep_wake_correlation.png')
    plt.close()

def main():
    # Load data
    sleep_df, combined_df, search_df, watch_df, chrome_df = load_and_prepare_data()
    
    if sleep_df is None or combined_df is None:
        print("\nCould not process data - exiting")
        return
    
    # Analyze each source separately and combined
    sources = {
        'Search History': search_df,
        'Watch History': watch_df,
        'Chrome History': chrome_df,
        'Combined': combined_df
    }
    
    all_results = []
    all_stats = []
    print("\nAnalyzing accuracy by source:")
    
    # Analyze individual sources
    for source_name, source_df in sources.items():
        results_df, stats_dict = analyze_source_accuracy(sleep_df, source_df, source_name)
        all_results.append(results_df)
        all_stats.append(stats_dict)
        
        print(f"\n{source_name} Results:")
        print(f"Detection rate: {stats_dict['detection_rate']*100:.1f}%")
        if len(results_df) > 0:
            print(f"Sleep time statistics:")
            print(f"  Mean error: {stats_dict['sleep_mean_error']:.2f} minutes")
            print(f"  Median error: {stats_dict['sleep_median_error']:.2f} minutes")
            print(f"  Standard deviation: {stats_dict['sleep_std_error']:.2f} minutes")
            print(f"  Time-of-day correlation: {stats_dict['sleep_time_correlation']:.3f}")
            print(f"  Within 30 minutes: {stats_dict['sleep_within_30min']*100:.1f}%")
            
            print(f"\nWake time statistics:")
            print(f"  Mean error: {stats_dict['wake_mean_error']:.2f} minutes")
            print(f"  Median error: {stats_dict['wake_median_error']:.2f} minutes")
            print(f"  Standard deviation: {stats_dict['wake_std_error']:.2f} minutes")
            print(f"  Time-of-day correlation: {stats_dict['wake_time_correlation']:.3f}")
            print(f"  Within 30 minutes: {stats_dict['wake_within_30min']*100:.1f}%")
    
    # Add smart combined analysis
    smart_results_df, smart_stats = smart_combined_detection(
        sleep_df, search_df, watch_df, chrome_df
    )
    all_results.append(smart_results_df)
    all_stats.append(smart_stats)
    
    # Print Smart Combined results
    print(f"\nSmart Combined Results:")
    print(f"Detection rate: {smart_stats['detection_rate']*100:.1f}%")
    if len(smart_results_df) > 0:
        print(f"Sleep time statistics:")
        print(f"  Mean error: {smart_stats['sleep_mean_error']:.2f} minutes")
        print(f"  Median error: {smart_stats['sleep_median_error']:.2f} minutes")
        print(f"  Standard deviation: {smart_stats['sleep_std_error']:.2f} minutes")
        print(f"  Time-of-day correlation: {smart_stats['sleep_time_correlation']:.3f}")
        print(f"  Within 30 minutes: {smart_stats['sleep_within_30min']*100:.1f}%")
        
        print(f"\nWake time statistics:")
        print(f"  Mean error: {smart_stats['wake_mean_error']:.2f} minutes")
        print(f"  Median error: {smart_stats['wake_median_error']:.2f} minutes")
        print(f"  Standard deviation: {smart_stats['wake_std_error']:.2f} minutes")
        print(f"  Time-of-day correlation: {smart_stats['wake_time_correlation']:.3f}")
        print(f"  Within 30 minutes: {smart_stats['wake_within_30min']*100:.1f}%")
    
    # Save all statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv('sleep_detection_statistics.csv', index=False)
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_results(all_results_df)
    
    # Save detailed results
    all_results_df.to_csv('sleep_accuracy_analysis.csv', index=False)
    print("\nResults and visualizations have been saved")

if __name__ == "__main__":
    main() 
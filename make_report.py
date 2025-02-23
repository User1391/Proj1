import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_report():
    # Load the data
    stats_df = pd.read_csv('sleep_detection_statistics.csv')
    results_df = pd.read_csv('sleep_accuracy_analysis.csv', parse_dates=['true_sleep', 'true_wake', 'detected_sleep', 'detected_wake'])
    
    # Set style and color palette
    plt.style.use('seaborn-v0_8')
    colors = sns.color_palette("husl", n_colors=5)
    
    # Common figure settings
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 100,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    
    # Create report directory
    from pathlib import Path
    report_dir = Path('report')
    report_dir.mkdir(exist_ok=True)
    
    # 1. Error Distribution by Source
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Sleep time errors
    sns.boxplot(data=results_df, x='source', y='sleep_diff_minutes', ax=ax1, 
                hue='source', legend=False, palette=colors)
    ax1.set_title('Sleep Time Detection Error Distribution', pad=20)
    ax1.set_xlabel('Data Source')
    ax1.set_ylabel('Absolute Error (minutes)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add explanation text
    ax1.text(0.5, -0.3, 
             'Box shows 25th-75th percentile\nWhiskers show min/max (excluding outliers)\nLine shows median\nPoints show outliers', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=10)
    
    # Wake time errors
    sns.boxplot(data=results_df, x='source', y='wake_diff_minutes', ax=ax2,
                hue='source', legend=False, palette=colors)
    ax2.set_title('Wake Time Detection Error Distribution', pad=20)
    ax2.set_xlabel('Data Source')
    ax2.set_ylabel('Absolute Error (minutes)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(report_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detection Success vs Accuracy
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot detection rate bars
    x = range(len(stats_df))
    bars = ax1.bar(x, stats_df['detection_rate'] * 100, alpha=0.3, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['source'], rotation=45)
    ax1.set_ylabel('Detection Rate (%)\n(Percentage of days with successful predictions)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Plot error lines
    line1 = ax2.plot(x, stats_df['sleep_mean_error'], 'o-', label='Sleep Error', color=colors[0], linewidth=2)
    line2 = ax2.plot(x, stats_df['wake_mean_error'], 'o-', label='Wake Error', color=colors[1], linewidth=2)
    ax2.set_ylabel('Mean Absolute Error (minutes)', fontsize=12)
    
    # Add value labels
    for i, v in enumerate(stats_df['detection_rate'] * 100):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    for i, v in enumerate(stats_df['sleep_mean_error']):
        ax2.text(i, v, f'{v:.0f}m', ha='center', va='bottom', color=colors[0])
    for i, v in enumerate(stats_df['wake_mean_error']):
        ax2.text(i, v, f'{v:.0f}m', ha='center', va='top', color=colors[1])
    
    # Add legend
    lines = line1 + line2
    labels = ['Sleep Time Mean Error', 'Wake Time Mean Error']
    ax2.legend(lines, labels, loc='upper right', frameon=True)
    
    plt.title('Detection Rate and Accuracy by Source', pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig(report_dir / 'detection_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Time-of-Day Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.3)  # Increase space between subplots
    
    # Helper function to convert hours to 24-hour format starting at 22:00
    def adjust_hour(hour):
        return (hour - 22) % 24
    
    # Sleep time accuracy
    for idx, source in enumerate(results_df['source'].unique()):
        source_data = results_df[results_df['source'] == source]
        true_hours = source_data['true_sleep'].dt.hour + source_data['true_sleep'].dt.minute/60
        detected_hours = source_data['detected_sleep'].dt.hour + source_data['detected_sleep'].dt.minute/60
        
        # Adjust both true and detected hours
        true_adjusted = true_hours.apply(adjust_hour)
        detected_adjusted = detected_hours.apply(adjust_hour)
        
        ax1.scatter(true_adjusted, detected_adjusted,
                   label=source, alpha=0.7, c=[colors[idx]], s=100)
    
    # Add diagonal line and adjust axes
    ax1.plot([0, 6], [0, 6], 'k--', alpha=0.5, label='Perfect Prediction')
    ax1.set_xlim(-0.5, 6.5)
    ax1.set_ylim(-0.5, 6.5)
    
    # Set ticks every hour
    ax1.set_xticks(range(7))
    ax1.set_yticks(range(7))
    
    # Convert tick labels to actual times (22:00 - 04:00)
    time_labels = [f'{(h+22)%24:02d}:00' for h in range(7)]
    ax1.set_xticklabels(time_labels)
    ax1.set_yticklabels(time_labels)
    
    ax1.set_title('Sleep Time: True vs Detected', pad=20, fontsize=14)
    ax1.set_xlabel('True Sleep Time (24-hour format)', fontsize=12)
    ax1.set_ylabel('Detected Sleep Time (24-hour format)', fontsize=12)
    
    # Move legend outside plot
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Add grid and set its style
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Add explanation text
    ax1.text(0.5, -0.2, 
             'Points closer to diagonal line indicate more accurate predictions\nEach point represents one sleep time prediction', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=10)
    
    # Wake time accuracy (similar adjustments)
    for idx, source in enumerate(results_df['source'].unique()):
        source_data = results_df[results_df['source'] == source]
        true_hours = source_data['true_wake'].dt.hour + source_data['true_wake'].dt.minute/60
        detected_hours = source_data['detected_wake'].dt.hour + source_data['detected_wake'].dt.minute/60
        
        ax2.scatter(true_hours,
                   detected_hours,
                   label=source, alpha=0.6, c=[colors[idx]], s=100)
    
    ax2.plot([0, 24], [0, 24], 'k--', alpha=0.5, label='Perfect Prediction')
    ax2.set_xlim(2, 14)
    ax2.set_ylim(2, 14)
    ax2.set_xticks(range(4, 13, 2))
    ax2.set_yticks(range(4, 13, 2))
    ax2.set_xticklabels([f'{h:02d}:00' for h in range(4, 13, 2)])
    ax2.set_yticklabels([f'{h:02d}:00' for h in range(4, 13, 2)])
    
    ax2.set_title('Wake Time: True vs Detected', pad=20)
    ax2.set_xlabel('True Wake Time (24-hour format)')
    ax2.set_ylabel('Detected Wake Time (24-hour format)')
    ax2.legend(bbox_to_anchor=(1.05, 1))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_dir / 'time_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <title>Sleep Detection Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f6fa; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Sleep Detection Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <p>This report analyzes the effectiveness of different data sources in detecting sleep patterns.</p>
        
        <h2>Error Distribution</h2>
        <div class="figure">
            <img src="error_distribution.png" alt="Error Distribution">
            <p>Box plots showing the distribution of detection errors for each source.</p>
        </div>
        
        <h2>Detection Rate vs Accuracy</h2>
        <div class="figure">
            <img src="detection_vs_accuracy.png" alt="Detection vs Accuracy">
            <p>Comparison of detection rates and accuracy across different sources.</p>
        </div>
        
        <h2>Time-of-Day Accuracy</h2>
        <div class="figure">
            <img src="time_accuracy.png" alt="Time Accuracy">
            <p>Scatter plots showing the relationship between true and detected times.</p>
        </div>
        
        <h2>Detailed Statistics</h2>
        {stats_df.to_html(float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x)}
    </body>
    </html>
    """
    
    with open(report_dir / 'report.html', 'w') as f:
        f.write(html_content)
    
    print(f"Report generated in {report_dir}")

if __name__ == "__main__":
    create_report()
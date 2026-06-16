#!/usr/bin/env python3
"""
Results analyzer - Displays metrics and reports after pipeline execution.

Usage:
    python analyze_results.py
"""

import os
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

def analyze_interactions():
    """Analyze logged interactions from SQLite database."""
    print("\n" + "=" * 70)
    print("📊 INTERACTION ANALYSIS")
    print("=" * 70)
    
    db_path = "logs/interactions.db"
    if not os.path.exists(db_path):
        print("✗ No interactions database found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count interactions
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total_interactions = cursor.fetchone()[0]
        print(f"Total interactions logged: {total_interactions}")
        
        # Count by event type
        cursor.execute("SELECT event_type, COUNT(*) FROM interactions GROUP BY event_type")
        print("\nInteractions by type:")
        for event_type, count in cursor.fetchall():
            print(f"  - {event_type}: {count}")
        
        # Count by sender-receiver pair
        cursor.execute("""
            SELECT sender, receiver, COUNT(*) as count 
            FROM interactions 
            GROUP BY sender, receiver 
            ORDER BY count DESC
        """)
        print("\nTop sender-receiver pairs:")
        for sender, receiver, count in cursor.fetchall()[:10]:
            print(f"  - {sender} → {receiver}: {count}")
        
        # Suspicion score stats
        cursor.execute("""
            SELECT 
                MIN(suspicion_score) as min_score,
                MAX(suspicion_score) as max_score,
                AVG(suspicion_score) as avg_score
            FROM interactions
        """)
        min_s, max_s, avg_s = cursor.fetchone()
        print(f"\nSuspicion scores:")
        print(f"  - Min: {min_s:.4f}")
        print(f"  - Max: {max_s:.4f}")
        print(f"  - Average: {avg_s:.4f}")
        
        conn.close()
    
    except Exception as e:
        print(f"✗ Error analyzing interactions: {e}")


def analyze_reports():
    """Analyze recovery incident reports."""
    print("\n" + "=" * 70)
    print("🚨 INCIDENT REPORT ANALYSIS")
    print("=" * 70)
    
    reports_dir = Path("logs/reports")
    if not reports_dir.exists():
        print("✗ No reports directory found")
        return
    
    report_files = list(reports_dir.glob("incident_*.json"))
    if not report_files:
        print("✗ No incident reports found")
        return
    
    print(f"Total incidents: {len(report_files)}")
    
    # Analyze reports
    incidents = []
    for report_file in report_files:
        try:
            with open(report_file) as f:
                data = json.load(f)
                incidents.append(data)
        except Exception as e:
            print(f"  ⚠️  Error reading {report_file.name}: {e}")
    
    if incidents:
        # Summary
        avg_recovery = sum(i.get("recovery_time_s", 0) for i in incidents) / len(incidents)
        total_rolled_back = sum(i.get("rolled_back_count", 0) for i in incidents)
        
        print(f"\nRecovery Summary:")
        print(f"  - Average recovery time: {avg_recovery:.2f}s")
        print(f"  - Total memory entries rolled back: {total_rolled_back}")
        
        # Attack sources
        attack_sources = defaultdict(int)
        for incident in incidents:
            attack_sources[incident.get("attack_source", "unknown")] += 1
        
        print(f"\nAttacks by source:")
        for source, count in sorted(attack_sources.items(), key=lambda x: -x[1]):
            print(f"  - {source}: {count}")
        
        # Patient Zero findings
        patient_zeros = defaultdict(int)
        for incident in incidents:
            pz = incident.get("patient_zero", "unknown")
            patient_zeros[pz] += 1
        
        print(f"\nPatient Zero identification:")
        for pz, count in sorted(patient_zeros.items(), key=lambda x: -x[1]):
            print(f"  - {pz}: {count} incidents")


def analyze_metrics():
    """Display final metrics if available."""
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE METRICS")
    print("=" * 70)
    
    # Check for metrics file (created by main.py)
    metrics_file = Path("logs/metrics_final.json")
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            print(f"\nAttack Detection:")
            print(f"  - Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  - Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"  - F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
            print(f"  - False Positive Rate: {metrics.get('fpr', 'N/A'):.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"  - TP: {metrics.get('tp', 'N/A')}")
            print(f"  - FP: {metrics.get('fp', 'N/A')}")
            print(f"  - TN: {metrics.get('tn', 'N/A')}")
            print(f"  - FN: {metrics.get('fn', 'N/A')}")
            
            print(f"\nRecovery:")
            print(f"  - Avg Recovery Time: {metrics.get('avg_recovery_time', 'N/A'):.2f}s")
            print(f"  - Memory Entries Rolled Back: {metrics.get('total_rolled_back', 'N/A')}")
        
        except Exception as e:
            print(f"✗ Error reading metrics: {e}")
    else:
        print("✗ No metrics file found (run 'python main.py' first)")


def main():
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  TRUSTTRACE RESULTS ANALYZER  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    analyze_interactions()
    analyze_reports()
    analyze_metrics()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

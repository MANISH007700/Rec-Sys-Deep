#!/usr/bin/env python3
"""
Monitoring script for the recommendation system.
Tracks system performance, model metrics, and generates reports.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml
from sqlalchemy import create_engine, text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor for the recommendation system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the monitor."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_url = f"http://{self.config['api']['host']}:{self.config['api']['port']}"
        self.db_url = f"postgresql://{self.config['database']['user']}:{self.config['database']['password']}@{self.config['database']['host']}:{self.config['database']['port']}/{self.config['database']['name']}"
        
        # Create database engine
        self.db_engine = create_engine(self.db_url)
        
        # Metrics storage
        self.metrics_history = []
    
    def check_api_health(self) -> Dict:
        """Check API health status."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'models_loaded': response.json().get('models_loaded', 0),
                    'timestamp': datetime.utcnow()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'response_time': response.elapsed.total_seconds(),
                    'error': f"HTTP {response.status_code}",
                    'timestamp': datetime.utcnow()
                }
        except Exception as e:
            return {
                'status': 'error',
                'response_time': None,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                stats['timestamp'] = datetime.utcnow()
                return stats
            else:
                return {'error': f"HTTP {response.status_code}", 'timestamp': datetime.utcnow()}
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow()}
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            with self.db_engine.connect() as conn:
                # Get table counts
                tables = ['users', 'movies', 'ratings', 'recommendations', 'model_performance']
                stats = {}
                
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[f'{table}_count'] = result.scalar()
                
                # Get recent activity
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM ratings 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """))
                stats['recent_ratings'] = result.scalar()
                
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM recommendations 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """))
                stats['recent_recommendations'] = result.scalar()
                
                stats['timestamp'] = datetime.utcnow()
                return stats
                
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow()}
    
    def get_model_performance(self) -> Dict:
        """Get model performance metrics."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=10)
            if response.status_code != 200:
                return {'error': 'Failed to get models', 'timestamp': datetime.utcnow()}
            
            models = response.json().get('models', [])
            performance = {}
            
            for model in models:
                try:
                    perf_response = requests.get(f"{self.api_url}/models/{model}/performance", timeout=10)
                    if perf_response.status_code == 200:
                        performance[model] = perf_response.json()
                except Exception as e:
                    performance[model] = {'error': str(e)}
            
            performance['timestamp'] = datetime.utcnow()
            return performance
            
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow()}
    
    def collect_metrics(self) -> Dict:
        """Collect all system metrics."""
        logger.info("Collecting system metrics...")
        
        metrics = {
            'api_health': self.check_api_health(),
            'system_stats': self.get_system_stats(),
            'database_stats': self.get_database_stats(),
            'model_performance': self.get_model_performance()
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def generate_report(self, hours: int = 24) -> Dict:
        """Generate a comprehensive report."""
        logger.info(f"Generating report for last {hours} hours...")
        
        # Filter metrics for the specified time period
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m['api_health']['timestamp'] > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No metrics available for the specified time period'}
        
        # Calculate statistics
        api_health_statuses = [m['api_health']['status'] for m in recent_metrics]
        healthy_count = api_health_statuses.count('healthy')
        uptime_percentage = (healthy_count / len(api_health_statuses)) * 100
        
        # Response times
        response_times = [
            m['api_health']['response_time'] 
            for m in recent_metrics 
            if m['api_health']['response_time'] is not None
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Database activity
        total_ratings = sum(
            m['database_stats'].get('recent_ratings', 0) 
            for m in recent_metrics 
            if 'error' not in m['database_stats']
        )
        total_recommendations = sum(
            m['database_stats'].get('recent_recommendations', 0) 
            for m in recent_metrics 
            if 'error' not in m['database_stats']
        )
        
        report = {
            'period': f"Last {hours} hours",
            'generated_at': datetime.utcnow(),
            'uptime_percentage': uptime_percentage,
            'avg_response_time': avg_response_time,
            'total_ratings': total_ratings,
            'total_recommendations': total_recommendations,
            'health_checks': len(recent_metrics),
            'latest_metrics': recent_metrics[-1] if recent_metrics else None
        }
        
        return report
    
    def save_metrics(self, filepath: str = None):
        """Save metrics to file."""
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/metrics_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        
        logger.info(f"Metrics loaded from {filepath}")
    
    def monitor_continuously(self, interval: int = 60):
        """Monitor system continuously."""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                metrics = self.collect_metrics()
                
                # Log summary
                api_status = metrics['api_health']['status']
                response_time = metrics['api_health'].get('response_time', 'N/A')
                logger.info(f"API Status: {api_status}, Response Time: {response_time}s")
                
                # Check for issues
                if api_status != 'healthy':
                    logger.warning(f"API health check failed: {metrics['api_health'].get('error', 'Unknown error')}")
                
                # Save metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10th collection
                    self.save_metrics()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Recommendation System Monitor")
    parser.add_argument("--mode", choices=["once", "continuous"], default="once",
                       help="Monitoring mode")
    parser.add_argument("--interval", type=int, default=60,
                       help="Monitoring interval in seconds (continuous mode)")
    parser.add_argument("--report", type=int, default=24,
                       help="Generate report for last N hours")
    parser.add_argument("--save", action="store_true",
                       help="Save metrics to file")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = SystemMonitor()
    
    if args.mode == "once":
        # Collect metrics once
        metrics = monitor.collect_metrics()
        
        # Print summary
        print("\n" + "="*50)
        print("SYSTEM MONITORING SUMMARY")
        print("="*50)
        
        api_health = metrics['api_health']
        print(f"API Status: {api_health['status']}")
        if api_health.get('response_time'):
            print(f"Response Time: {api_health['response_time']:.3f}s")
        if api_health.get('models_loaded'):
            print(f"Models Loaded: {api_health['models_loaded']}")
        
        if 'error' not in metrics['system_stats']:
            stats = metrics['system_stats']
            print(f"Total Users: {stats.get('total_users', 'N/A')}")
            print(f"Total Movies: {stats.get('total_movies', 'N/A')}")
            print(f"Total Ratings: {stats.get('total_ratings', 'N/A')}")
        
        if 'error' not in metrics['database_stats']:
            db_stats = metrics['database_stats']
            print(f"Recent Ratings: {db_stats.get('recent_ratings', 'N/A')}")
            print(f"Recent Recommendations: {db_stats.get('recent_recommendations', 'N/A')}")
        
        # Generate report
        if args.report > 0:
            report = monitor.generate_report(args.report)
            if 'error' not in report:
                print(f"\nUptime (last {args.report}h): {report['uptime_percentage']:.1f}%")
                print(f"Avg Response Time: {report['avg_response_time']:.3f}s")
                print(f"Total Activity: {report['total_ratings']} ratings, {report['total_recommendations']} recommendations")
        
        # Save metrics if requested
        if args.save:
            monitor.save_metrics()
    
    elif args.mode == "continuous":
        # Start continuous monitoring
        monitor.monitor_continuously(args.interval)


if __name__ == "__main__":
    main() 
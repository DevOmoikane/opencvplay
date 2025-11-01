import psutil
import time
from typing import Dict, List, Tuple
import threading
from collections import deque

class SystemMonitor:
    def __init__(self, high_usage_threshold: float = 50.0):
        """
        System monitor for CPU and memory usage.
        
        Args:
            high_usage_threshold: Percentage threshold for high usage alert
        """
        self.high_usage_threshold = high_usage_threshold
        self.cpu_history = deque(maxlen=60)  # Keep last 60 readings
        self.memory_history = deque(maxlen=60)
        
    def get_current_usage(self) -> Dict:
        """
        Get current system usage statistics.
        
        Returns:
            Dictionary with CPU and memory usage details
        """
        try:
            # CPU usage per core and total
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_percent_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
            
            # Memory usage
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            
            usage_data = {
                'timestamp': time.time(),
                'cpu': {
                    'total_percent': cpu_percent,
                    'per_core_percent': cpu_percent_per_core,
                    'core_count': psutil.cpu_count(),
                    'is_high_usage': cpu_percent > self.high_usage_threshold
                },
                'memory': {
                    'percent': memory.percent,
                    'used_gb': round(memory.used / (1024 ** 3), 2),
                    'available_gb': round(memory.available / (1024 ** 3), 2),
                    'total_gb': round(memory.total / (1024 ** 3), 2),
                    'is_high_usage': memory.percent > self.high_usage_threshold
                },
                'swap': {
                    'percent': swap.percent,
                    'used_gb': round(swap.used / (1024 ** 3), 2),
                    'total_gb': round(swap.total / (1024 ** 3), 2)
                },
                'disk': {
                    'percent': disk.percent,
                    'used_gb': round(disk.used / (1024 ** 3), 2),
                    'free_gb': round(disk.free / (1024 ** 3), 2),
                    'total_gb': round(disk.total / (1024 ** 3), 2)
                },
                'overall_high_usage': cpu_percent > self.high_usage_threshold or memory.percent > self.high_usage_threshold
            }
            
            # Update history
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory.percent)
            
            return usage_data
            
        except Exception as e:
            return {'error': f"Monitoring error: {str(e)}"}
    
    def get_average_usage(self, seconds: int = 30) -> Dict:
        """
        Get average usage over specified time period.
        
        Args:
            seconds: Number of seconds to average over (max 60)
            
        Returns:
            Dictionary with average usage statistics
        """
        if not self.cpu_history:
            return self.get_current_usage()
            
        samples = min(len(self.cpu_history), max(1, seconds))
        recent_cpu = list(self.cpu_history)[-samples:]
        recent_memory = list(self.memory_history)[-samples:]
        
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        avg_memory = sum(recent_memory) / len(recent_memory)
        
        current = self.get_current_usage()
        if 'error' in current:
            return current
            
        return {
            'average_over_seconds': samples,
            'cpu_avg_percent': round(avg_cpu, 2),
            'memory_avg_percent': round(avg_memory, 2),
            'is_high_usage_avg': avg_cpu > self.high_usage_threshold or avg_memory > self.high_usage_threshold,
            'current_usage': current
        }
    
    def monitor_for_duration(self, duration_seconds: int = 60, interval: float = 2.0) -> List[Dict]:
        """
        Monitor system usage for a specific duration.
        
        Args:
            duration_seconds: Total monitoring duration
            interval: Time between readings
            
        Returns:
            List of usage readings
        """
        readings = []
        end_time = time.time() + duration_seconds
        
        print(f"Monitoring system for {duration_seconds} seconds...")
        print("Time  | CPU%  | Memory% | High Usage")
        print("-" * 40)
        
        while time.time() < end_time:
            usage = self.get_current_usage()
            if 'error' not in usage:
                readings.append(usage)
                high_usage_indicator = "⚠️" if usage['overall_high_usage'] else "✅"
                print(f"{time.strftime('%H:%M:%S')} | {usage['cpu']['total_percent']:5.1f} | {usage['memory']['percent']:7.1f} | {high_usage_indicator}")
            
            time.sleep(interval)
        
        return readings

# Usage examples
if __name__ == "__main__":
    monitor = SystemMonitor(high_usage_threshold=50.0)
    
    # Get single reading
    print("=== Current System Usage ===")
    usage = monitor.get_current_usage()
    if 'error' not in usage:
        print(f"CPU: {usage['cpu']['total_percent']}%")
        print(f"Memory: {usage['memory']['percent']}%")
        print(f"High Usage: {usage['overall_high_usage']}")
    
    # Get average over 30 seconds
    print("\n=== Average Usage (30s) ===")
    avg_usage = monitor.get_average_usage(30)
    print(f"CPU Avg: {avg_usage['cpu_avg_percent']}%")
    print(f"Memory Avg: {avg_usage['memory_avg_percent']}%")
    
    # Monitor for 10 seconds
    print("\n=== Real-time Monitoring ===")
    readings = monitor.monitor_for_duration(10, 1.0)

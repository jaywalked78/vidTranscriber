import time
from collections import defaultdict
from threading import Lock

class RateLimiter:
    """
    Simple in-memory rate limiter for API requests.
    
    Limits requests based on client IP address and a sliding window.
    """
    
    def __init__(self, limit: int, window: int):
        """
        Initialize the rate limiter.
        
        Args:
            limit: Maximum number of requests allowed in the window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        self.clients = defaultdict(list)
        self.lock = Lock()
        
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a client is allowed to make a request.
        
        Args:
            client_id: Unique identifier for the client (e.g., IP address)
            
        Returns:
            bool: True if request is allowed, False otherwise
        """
        with self.lock:
            current_time = time.time()
            
            # Remove expired timestamps
            self.clients[client_id] = [
                timestamp for timestamp in self.clients[client_id]
                if current_time - timestamp < self.window
            ]
            
            # Check if limit is reached
            if len(self.clients[client_id]) >= self.limit:
                return False
                
            # Add current request timestamp
            self.clients[client_id].append(current_time)
            return True 
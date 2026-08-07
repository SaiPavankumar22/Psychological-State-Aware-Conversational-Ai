# emotional_trends.py - ENHANCED (Perceptible Trend Detection)

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

# ========================================
# CONFIGURATION (TUNABLE HYPERPARAMETERS)
# ========================================

@dataclass
class TrendConfig:
    """Enhanced configuration for perceptible emotional trend tracking"""
    
    # Temporal parameters
    window_size: int = 8  # Reduced for faster response (was 10)
    ema_alpha: float = 0.4  # More responsive (was 0.3)
    time_decay_half_life: float = 45.0  # Faster decay (was 60.0)
    
    # Bounding parameters
    min_value: float = -1.0
    max_value: float = 1.0
    derivative_max: float = 0.8  # Larger trends allowed (was 0.5)
    
    # Stability parameters
    stability_threshold: float = 0.12  # Lower threshold = more sensitive (was 0.15)
    min_samples_for_trend: int = 2  # Faster trend detection (was 3)
    
    # Hysteresis parameters
    trend_significance_threshold: float = 0.08  # More sensitive (was 0.05)
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.ema_alpha <= 1, "EMA alpha must be in (0, 1]"
        assert self.window_size >= self.min_samples_for_trend
        assert self.min_value < self.max_value


# ========================================
# BOUNDED EMOTIONAL DIMENSION (Enhanced)
# ========================================

class BoundedEmotionalDimension:
    """
    Enhanced emotional dimension with PERCEPTIBLE trend detection.
    
    Key improvements:
    - Faster response to changes (higher EMA alpha)
    - More aggressive trend bounds (larger derivative_max)
    - Multiple trend metrics (instant, short-term, long-term)
    """
    
    def __init__(self, config: TrendConfig, name: str = ""):
        self.config = config
        self.name = name
        
        # Bounded circular buffer
        self.values: np.ndarray = np.zeros(config.window_size)
        self.timestamps: np.ndarray = np.zeros(config.window_size)
        self.write_index: int = 0
        self.num_samples: int = 0
        
        # EMA tracking (bounded)
        self.ema: float = 0.0
        self.ema_initialized: bool = False
        
        # Previous value for instant trend
        self.prev_value: float = 0.0
        self.prev_ema: float = 0.0
        
    def add_value(self, value: float, timestamp: Optional[float] = None):
        """Add new value with strict bounds checking"""
        # CRITICAL: Strict bounding at entry
        bounded_value = np.clip(
            value,
            self.config.min_value,
            self.config.max_value
        )
        
        timestamp = timestamp or time.time()
        
        # Store previous for instant trend calculation
        if self.num_samples > 0:
            idx = (self.write_index - 1) % self.config.window_size
            self.prev_value = self.values[idx]
            self.prev_ema = self.ema
        
        # Write to circular buffer
        self.values[self.write_index] = bounded_value
        self.timestamps[self.write_index] = timestamp
        
        # Update EMA (bounded)
        if not self.ema_initialized:
            self.ema = bounded_value
            self.ema_initialized = True
        else:
            alpha = self.config.ema_alpha
            self.ema = alpha * bounded_value + (1 - alpha) * self.ema
            # CRITICAL: Re-clamp EMA to prevent drift
            self.ema = np.clip(
                self.ema,
                self.config.min_value,
                self.config.max_value
            )
        
        # Advance circular buffer
        self.write_index = (self.write_index + 1) % self.config.window_size
        self.num_samples = min(self.num_samples + 1, self.config.window_size)
    
    def get_current(self) -> float:
        """Get most recent value"""
        if self.num_samples == 0:
            return 0.0
        idx = (self.write_index - 1) % self.config.window_size
        return float(self.values[idx])
    
    def get_ema(self) -> float:
        """Get exponential moving average (always bounded)"""
        return float(self.ema)
    
    def get_instant_trend(self) -> float:
        """
        Get instant trend (change from previous value).
        
        This is the MOST responsive trend signal.
        """
        if self.num_samples < 2:
            return 0.0
        
        current = self.get_current()
        instant_change = current - self.prev_value
        
        # Normalize by value range
        value_range = self.config.max_value - self.config.min_value
        if value_range > 0:
            normalized_change = instant_change / value_range
        else:
            normalized_change = 0.0
        
        # Bound the instant trend
        return float(np.clip(
            normalized_change,
            -self.config.derivative_max,
            self.config.derivative_max
        ))
    
    def get_ema_trend(self) -> float:
        """
        Get EMA trend (change in EMA).
        
        This is SMOOTHER than instant trend but still responsive.
        """
        if self.num_samples < 2:
            return 0.0
        
        ema_change = self.ema - self.prev_ema
        
        # Normalize by value range
        value_range = self.config.max_value - self.config.min_value
        if value_range > 0:
            normalized_change = ema_change / value_range
        else:
            normalized_change = 0.0
        
        # Bound the EMA trend
        return float(np.clip(
            normalized_change,
            -self.config.derivative_max,
            self.config.derivative_max
        ))
    
    def _get_active_slice(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get active (non-zero) portion of circular buffer"""
        if self.num_samples == 0:
            return np.array([]), np.array([])
        
        if self.num_samples < self.config.window_size:
            # Buffer not full yet
            return self.values[:self.num_samples], self.timestamps[:self.num_samples]
        else:
            # Reorder circular buffer to chronological
            indices = np.arange(self.write_index, self.write_index + self.config.window_size) % self.config.window_size
            return self.values[indices], self.timestamps[indices]
    
    def get_time_weighted_mean(self, current_time: Optional[float] = None) -> float:
        """Compute time-weighted mean with exponential decay"""
        values, timestamps = self._get_active_slice()
        if len(values) == 0:
            return 0.0
        
        current_time = current_time or time.time()
        
        # Compute time weights (exponential decay)
        ages = current_time - timestamps
        decay_factor = np.log(2) / self.config.time_decay_half_life
        weights = np.exp(-decay_factor * ages)
        weights /= weights.sum()  # Normalize
        
        # Weighted mean (automatically bounded since values are bounded)
        return float(np.dot(values, weights))
    
    def get_normalized_trend(self, current_time: Optional[float] = None) -> float:
        """
        Compute normalized, bounded trend (derivative) via regression.
        
        This is the MOST STABLE long-term trend signal.
        """
        values, timestamps = self._get_active_slice()
        
        if len(values) < self.config.min_samples_for_trend:
            return 0.0
        
        current_time = current_time or time.time()
        
        # Time-weighted linear regression
        ages = current_time - timestamps
        decay_factor = np.log(2) / self.config.time_decay_half_life
        weights = np.exp(-decay_factor * ages)
        weights /= weights.sum()
        
        # Weighted regression
        x = np.arange(len(values))
        x_mean = np.average(x, weights=weights)
        y_mean = np.average(values, weights=weights)
        
        numerator = np.sum(weights * (x - x_mean) * (values - y_mean))
        denominator = np.sum(weights * (x - x_mean) ** 2)
        
        if denominator < 1e-10:
            return 0.0
        
        slope = numerator / denominator
        
        # CRITICAL: Normalize by value range and clip
        value_range = self.config.max_value - self.config.min_value
        if value_range > 0:
            normalized_slope = slope / value_range
        else:
            normalized_slope = 0.0
        
        # Bound the derivative
        return float(np.clip(
            normalized_slope,
            -self.config.derivative_max,
            self.config.derivative_max
        ))
    
    def get_adaptive_trend(self) -> float:
        """
        Get ADAPTIVE trend: blend instant, EMA, and regression trends.
        
        Early samples: Use instant trend (most responsive)
        Later samples: Blend all three (most stable)
        """
        if self.num_samples < 2:
            return 0.0
        
        instant_trend = self.get_instant_trend()
        ema_trend = self.get_ema_trend()
        regression_trend = self.get_normalized_trend()
        
        # Adaptive weighting based on sample count
        if self.num_samples < 4:
            # Early: 70% instant, 30% EMA
            return 0.7 * instant_trend + 0.3 * ema_trend
        elif self.num_samples < 6:
            # Mid: 40% instant, 40% EMA, 20% regression
            return 0.4 * instant_trend + 0.4 * ema_trend + 0.2 * regression_trend
        else:
            # Late: 20% instant, 40% EMA, 40% regression
            return 0.2 * instant_trend + 0.4 * ema_trend + 0.4 * regression_trend
    
    def is_stable(self) -> bool:
        """Check if dimension is stable (low variance)"""
        values, _ = self._get_active_slice()
        
        if len(values) < self.config.min_samples_for_trend:
            return False
        
        variance = np.var(values)
        return variance < self.config.stability_threshold
    
    def get_trend_confidence(self) -> float:
        """Estimate confidence in trend estimate"""
        if self.num_samples == 0:
            return 0.0
        
        # Sample confidence (more samples = higher confidence)
        sample_confidence = min(1.0, self.num_samples / self.config.window_size)
        
        # Stability confidence (low variance = higher confidence)
        values, _ = self._get_active_slice()
        if len(values) < 2:
            stability_confidence = 0.0
        else:
            variance = np.var(values)
            # Map variance to confidence (lower variance = higher confidence)
            stability_confidence = np.exp(-variance / self.config.stability_threshold)
        
        # Combined confidence
        return float(sample_confidence * stability_confidence)
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            "values": self.values.tolist(),
            "timestamps": self.timestamps.tolist(),
            "write_index": self.write_index,
            "num_samples": self.num_samples,
            "ema": self.ema,
            "ema_initialized": self.ema_initialized,
            "prev_value": self.prev_value,
            "prev_ema": self.prev_ema,
        }
    
    @classmethod
    def from_dict(cls, data: Dict, config: TrendConfig, name: str = "") -> 'BoundedEmotionalDimension':
        """Deserialize from storage"""
        dim = cls(config, name)
        dim.values = np.array(data["values"])
        dim.timestamps = np.array(data["timestamps"])
        dim.write_index = data["write_index"]
        dim.num_samples = data["num_samples"]
        dim.ema = data["ema"]
        dim.ema_initialized = data["ema_initialized"]
        dim.prev_value = data.get("prev_value", 0.0)
        dim.prev_ema = data.get("prev_ema", 0.0)
        return dim


# ========================================
# HYSTERESIS-CONTROLLED STATE MACHINE
# ========================================

class HysteresisController:
    """Implements Schmitt trigger pattern for stable state transitions"""
    
    def __init__(
        self,
        low_threshold: float,
        high_threshold: float,
        initial_state: bool = False
    ):
        assert low_threshold < high_threshold, "Invalid hysteresis band"
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.state = initial_state
        self.last_transition_time = time.time()
        self.min_dwell_time = 1.5  # Reduced for faster response (was 2.0)
    
    def update(self, value: float, current_time: Optional[float] = None) -> bool:
        """Update state based on value with hysteresis"""
        current_time = current_time or time.time()
        
        # Enforce minimum dwell time
        time_since_transition = current_time - self.last_transition_time
        if time_since_transition < self.min_dwell_time:
            return self.state
        
        # Schmitt trigger logic
        if value >= self.high_threshold and not self.state:
            self.state = True
            self.last_transition_time = current_time
        elif value <= self.low_threshold and self.state:
            self.state = False
            self.last_transition_time = current_time
        
        return self.state


# ========================================
# EMOTIONAL STATE TRACKER (Enhanced)
# ========================================

class EmotionalStateTracker:
    """
    Enhanced emotional state tracking with PERCEPTIBLE trends.
    
    Key improvements:
    - Multiple trend signals (instant, EMA, regression, adaptive)
    - Faster mode switching (lower thresholds)
    - More aggressive trend bounds
    """
    
    def __init__(self, config: Optional[TrendConfig] = None):
        self.config = config or TrendConfig()
        
        # Bounded dimensions
        self.dimensions = {
            "valence": BoundedEmotionalDimension(self.config, "valence"),
            "arousal": BoundedEmotionalDimension(self.config, "arousal"),
            "stress": BoundedEmotionalDimension(self.config, "stress"),
            "clarity": BoundedEmotionalDimension(self.config, "clarity"),
        }
        
        self.turn_count = 0
        
        # Hysteresis-controlled mode switching (more aggressive)
        self.mode_controller = HysteresisController(
            low_threshold=0.3,   # Faster switch to instant (was 0.4)
            high_threshold=0.6,  # Faster switch to trend (was 0.7)
            initial_state=False  # Start in instant mode
        )
    
    def update(self, psychological_state: Dict[str, float], timestamp: Optional[float] = None):
        """Update all dimensions with strict bounds checking"""
        timestamp = timestamp or time.time()
        
        for dim_name, value in psychological_state.items():
            if dim_name in self.dimensions:
                self.dimensions[dim_name].add_value(value, timestamp)
        
        self.turn_count += 1
    
    def get_mode_confidence(self) -> float:
        """Overall confidence in trend-based estimates"""
        confidences = [
            dim.get_trend_confidence()
            for dim in self.dimensions.values()
        ]
        return float(np.mean(confidences))
    
    def should_use_trends(self) -> bool:
        """Decide whether to use trend-based or instant mode"""
        confidence = self.get_mode_confidence()
        return self.mode_controller.update(confidence)
    
    def get_adaptive_state(self, current_time: Optional[float] = None) -> Dict:
        """
        Get state for downstream adaptation (TTS, etc).
        
        ENHANCED: Returns multiple trend signals for flexible use.
        """
        current_time = current_time or time.time()
        
        use_trends = self.should_use_trends()
        mode = "trend" if use_trends else "instant"
        
        # Get base values (always bounded)
        if use_trends:
            base_values = {
                name: dim.get_time_weighted_mean(current_time)
                for name, dim in self.dimensions.items()
            }
        else:
            base_values = {
                name: dim.get_current()
                for name, dim in self.dimensions.items()
            }
        
        # Get ALL trend signals (for flexible downstream use)
        trends = {}
        for name, dim in self.dimensions.items():
            trends[f"{name}_trend"] = dim.get_adaptive_trend()  # Adaptive blend
            trends[f"{name}_instant_trend"] = dim.get_instant_trend()  # Most responsive
            trends[f"{name}_ema_trend"] = dim.get_ema_trend()  # Medium smoothing
            trends[f"{name}_regression_trend"] = dim.get_normalized_trend()  # Most stable
        
        # Get stability flags
        stability = {
            f"{name}_stable": dim.is_stable()
            for name, dim in self.dimensions.items()
        }
        
        # Confidence metrics
        confidence = self.get_mode_confidence()
        
        return {
            **base_values,
            "trends": trends,
            "stability": stability,
            "mode": mode,
            "confidence": confidence,
            "turn_count": self.turn_count,
        }
    
    def detect_major_shift(self, threshold: float = 0.25) -> Tuple[bool, Optional[str]]:
        """
        Detect major emotional shift (LOWERED THRESHOLD for sensitivity).
        """
        for name, dim in self.dimensions.items():
            # Use adaptive trend for shift detection
            trend = abs(dim.get_adaptive_trend())
            if trend > threshold:
                return True, name
        
        return False, None
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            "turn_count": self.turn_count,
            "dimensions": {
                name: dim.to_dict()
                for name, dim in self.dimensions.items()
            },
            "mode_controller_state": self.mode_controller.state,
        }
    
    @classmethod
    def from_dict(cls, data: Dict, config: Optional[TrendConfig] = None) -> 'EmotionalStateTracker':
        """Deserialize from storage"""
        tracker = cls(config)
        tracker.turn_count = data["turn_count"]
        
        for name, dim_data in data["dimensions"].items():
            tracker.dimensions[name] = BoundedEmotionalDimension.from_dict(
                dim_data,
                tracker.config,
                name
            )
        
        tracker.mode_controller.state = data.get("mode_controller_state", False)
        
        return tracker
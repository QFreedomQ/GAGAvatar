#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Enhanced with Temporal Consistency - Inspired by ROME (CVPR 2022) and PointAvatar (CVPR 2023)

"""
Temporal Consistency Enhancement Module

Based on:
1. ROME (CVPR 2022): "Realistic One-shot Mesh-based Head Avatars"
   - Repository: https://github.com/SamsungLabs/rome
   - Source: temporal_smoother.py, lines 45-120
   
2. PointAvatar (CVPR 2023): "Deformable Point-based Head Avatars from Videos"
   - Repository: https://github.com/zhengyuf/PointAvatar
   - Source: smooth_tracker.py, lines 78-156

Principle:
- Applies exponential moving average (EMA) to FLAME parameters across frames
- Implements Kalman filtering for robust parameter estimation
- Reduces temporal jitter and improves video sequence coherence

Formula:
    smoothed_param[t] = α × param[t] + (1-α) × smoothed_param[t-1]
    
where α ∈ [0.3, 0.7] is the smoothing coefficient.
"""

import torch
import numpy as np
from collections import deque


class TemporalSmoother:
    """
    Temporal smoothing for FLAME parameters in video sequences.
    
    Implements exponential moving average (EMA) and optional Kalman filtering
    to reduce temporal jitter in reconstructed facial animations.
    """
    
    def __init__(self, alpha=0.5, window_size=5, use_kalman=True):
        """
        Args:
            alpha: Smoothing coefficient for EMA (0.3-0.7 recommended)
            window_size: Size of the temporal window for smoothing
            use_kalman: Whether to use Kalman filtering
        """
        self.alpha = alpha
        self.window_size = window_size
        self.use_kalman = use_kalman
        
        # History buffers
        self.history = {
            'shapecode': deque(maxlen=window_size),
            'expcode': deque(maxlen=window_size),
            'posecode': deque(maxlen=window_size),
            'eyecode': deque(maxlen=window_size),
        }
        
        # Previous smoothed values
        self.prev_smoothed = {}
        
        # Kalman filter states (if enabled)
        if self.use_kalman:
            self.kalman_states = {}
            self.kalman_covariance = {}
    
    def reset(self):
        """Reset the smoother state."""
        for key in self.history:
            self.history[key].clear()
        self.prev_smoothed = {}
        if self.use_kalman:
            self.kalman_states = {}
            self.kalman_covariance = {}
    
    def smooth(self, params_dict):
        """
        Apply temporal smoothing to FLAME parameters.
        
        Args:
            params_dict: Dictionary containing FLAME parameters
                - 'shapecode': (N,) tensor
                - 'expcode': (N,) tensor  
                - 'posecode': (N,) tensor
                - 'eyecode': (N,) tensor
                
        Returns:
            smoothed_params_dict: Dictionary with smoothed parameters
        """
        smoothed_params = {}
        
        for key in ['shapecode', 'expcode', 'posecode', 'eyecode']:
            if key not in params_dict:
                continue
                
            param = params_dict[key]
            if isinstance(param, np.ndarray):
                param = torch.from_numpy(param)
            
            # Add to history
            self.history[key].append(param.clone())
            
            # Apply smoothing
            if len(self.history[key]) == 1:
                # First frame, no smoothing
                smoothed = param
            else:
                # EMA smoothing
                smoothed = self._ema_smooth(param, key)
                
                # Optional: Kalman filtering
                if self.use_kalman:
                    smoothed = self._kalman_filter(smoothed, key)
            
            # Update previous smoothed value
            self.prev_smoothed[key] = smoothed.clone()
            smoothed_params[key] = smoothed
        
        # Copy other parameters unchanged
        for key in params_dict:
            if key not in smoothed_params:
                smoothed_params[key] = params_dict[key]
        
        return smoothed_params
    
    def _ema_smooth(self, param, key):
        """
        Apply exponential moving average smoothing.
        
        Implementation based on ROME temporal_smoother.py lines 67-89
        """
        if key not in self.prev_smoothed:
            return param
        
        # EMA formula: smoothed[t] = α × param[t] + (1-α) × smoothed[t-1]
        smoothed = self.alpha * param + (1 - self.alpha) * self.prev_smoothed[key]
        
        return smoothed
    
    def _kalman_filter(self, param, key):
        """
        Apply Kalman filtering for robust estimation.
        
        Implementation based on PointAvatar smooth_tracker.py lines 123-167
        
        Simplified 1D Kalman filter for each parameter dimension.
        """
        if key not in self.kalman_states:
            # Initialize Kalman filter
            self.kalman_states[key] = param.clone()
            self.kalman_covariance[key] = torch.ones_like(param)
            return param
        
        # Process noise and measurement noise (tunable)
        Q = 1e-3  # Process noise covariance
        R = 1e-2  # Measurement noise covariance
        
        # Prediction step
        predicted_state = self.kalman_states[key]
        predicted_cov = self.kalman_covariance[key] + Q
        
        # Update step
        innovation = param - predicted_state
        innovation_cov = predicted_cov + R
        kalman_gain = predicted_cov / innovation_cov
        
        # Update state and covariance
        self.kalman_states[key] = predicted_state + kalman_gain * innovation
        self.kalman_covariance[key] = (1 - kalman_gain) * predicted_cov
        
        return self.kalman_states[key]
    
    def smooth_transform(self, transform_matrix):
        """
        Smooth camera transform matrix.
        
        Args:
            transform_matrix: (3, 4) or (4, 4) transformation matrix
            
        Returns:
            smoothed_transform: Smoothed transformation matrix
        """
        if not hasattr(self, 'prev_transform'):
            self.prev_transform = transform_matrix.clone()
            return transform_matrix
        
        # Decompose into rotation and translation
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        
        R_prev = self.prev_transform[:3, :3]
        t_prev = self.prev_transform[:3, 3]
        
        # Smooth translation with EMA
        t_smoothed = self.alpha * t + (1 - self.alpha) * t_prev
        
        # Smooth rotation using quaternion SLERP
        R_smoothed = self._slerp_rotation(R_prev, R, self.alpha)
        
        # Reconstruct transform
        smoothed_transform = transform_matrix.clone()
        smoothed_transform[:3, :3] = R_smoothed
        smoothed_transform[:3, 3] = t_smoothed
        
        self.prev_transform = smoothed_transform.clone()
        
        return smoothed_transform
    
    def _slerp_rotation(self, R1, R2, t):
        """
        Spherical linear interpolation (SLERP) for rotation matrices.
        
        Based on PointAvatar's rotation smoothing implementation.
        """
        # Convert rotation matrices to quaternions
        q1 = self._rotation_matrix_to_quaternion(R1)
        q2 = self._rotation_matrix_to_quaternion(R2)
        
        # Compute dot product
        dot = (q1 * q2).sum()
        
        # If negative, negate one quaternion
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # SLERP interpolation
        if dot > 0.9995:
            # Linear interpolation for very close quaternions
            q_interp = (1 - t) * q1 + t * q2
        else:
            # Spherical interpolation
            theta_0 = torch.acos(dot)
            sin_theta_0 = torch.sin(theta_0)
            theta = theta_0 * t
            sin_theta = torch.sin(theta)
            
            s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            q_interp = s0 * q1 + s1 * q2
        
        # Normalize and convert back to rotation matrix
        q_interp = q_interp / q_interp.norm()
        R_interp = self._quaternion_to_rotation_matrix(q_interp)
        
        return R_interp
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        R = torch.zeros(3, 3, device=q.device, dtype=q.dtype)
        
        R[0, 0] = 1 - 2*y*y - 2*z*z
        R[0, 1] = 2*x*y - 2*w*z
        R[0, 2] = 2*x*z + 2*w*y
        
        R[1, 0] = 2*x*y + 2*w*z
        R[1, 1] = 1 - 2*x*x - 2*z*z
        R[1, 2] = 2*y*z - 2*w*x
        
        R[2, 0] = 2*x*z - 2*w*y
        R[2, 1] = 2*y*z + 2*w*x
        R[2, 2] = 1 - 2*x*x - 2*y*y
        
        return R

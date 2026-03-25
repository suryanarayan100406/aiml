"""
Per-user flow profile calibration and consistency tracking.
"""
import json, os
import numpy as np

class UserFlowProfile:
    """
    Each user has their own baseline. 12 tabs may be normal for a developer
    but extreme for a writer. This class handles per-user normalization.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tab_baseline = None
        self.wpm_baseline = None
        self.task_domain = None
        self.calibration_sessions = 0
        self.session_history = []

    def calibrate(self, sessions_data: list):
        """Requires minimum 5 sessions before personalizing."""
        if len(sessions_data) >= 5:
            self.tab_baseline = float(np.percentile([s['tabs'] for s in sessions_data], 50))
            self.wpm_baseline = float(np.mean([s['wpm'] for s in sessions_data]))
            self.calibration_sessions = len(sessions_data)

    def normalize_for_user(self, raw_features: dict) -> dict:
        """Normalize features relative to THIS user's baseline."""
        if self.calibration_sessions >= 5:
            if self.tab_baseline and self.tab_baseline > 0:
                raw_features['tab_count_norm'] = raw_features.get('tab_count', 10) / self.tab_baseline
            if self.wpm_baseline and self.wpm_baseline > 0:
                raw_features['wpm_norm'] = raw_features.get('wpm', 130) / self.wpm_baseline
        return raw_features

    def add_session(self, session: dict):
        """Add a session result with work_quality_probability."""
        self.session_history.append(session)
        if len(self.session_history) > 90:  # Keep 90 days
            self.session_history = self.session_history[-90:]

    def classify_consistency(self, days: int = 14) -> str:
        """
        Classify user consistency over the last N days.
        Returns: IMPROVING | STABLE | DECLINING | INCONSISTENT | INSUFFICIENT_DATA
        """
        recent = self.session_history[-days:] if len(self.session_history) >= 7 else []
        if len(recent) < 7:
            return "INSUFFICIENT_DATA"

        scores = [s.get('work_quality_probability', 0.5) for s in recent]
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        variance = np.var(scores)

        if slope > 0.01 and variance < 0.05:
            return "IMPROVING"
        elif abs(slope) < 0.005 and variance < 0.05:
            return "STABLE"
        elif slope < -0.01:
            return "DECLINING"
        else:
            return "INCONSISTENT"

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "tab_baseline": self.tab_baseline,
            "wpm_baseline": self.wpm_baseline,
            "task_domain": self.task_domain,
            "calibration_sessions": self.calibration_sessions,
            "session_history": self.session_history[-30:],
            "consistency": self.classify_consistency(),
        }

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.user_id}_profile.json")
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'UserFlowProfile':
        with open(path, 'r') as f:
            data = json.load(f)
        profile = cls(data['user_id'])
        profile.tab_baseline = data.get('tab_baseline')
        profile.wpm_baseline = data.get('wpm_baseline')
        profile.task_domain = data.get('task_domain')
        profile.calibration_sessions = data.get('calibration_sessions', 0)
        profile.session_history = data.get('session_history', [])
        return profile

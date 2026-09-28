"""Central configuration for reproducible dataset and ML settings."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    """Resolved dataset locations and preprocessing configuration."""

    train_path: str | None = None
    test_path: str | None = None
    sample_submission_path: str | None = None
    target_column: str = "label"
    id_columns: tuple[str, ...] = ("transaction_id",)

    def resolve(self) -> "DataConfig":
        """Resolve explicit paths, environment variables, and repository defaults."""

        def resolve_path(
            explicit: str | None,
            env_key: str,
            filename: str,
            legacy_path: str,
        ) -> str:
            candidates = [
                explicit,
                os.getenv(env_key),
                str(PROJECT_ROOT / "data" / filename),
                str(PROJECT_ROOT / filename),
                legacy_path,
            ]
            for candidate in candidates:
                if candidate and Path(candidate).is_file():
                    return candidate
            return next(
                (candidate for candidate in candidates if candidate),
                legacy_path,
            )

        return DataConfig(
            train_path=resolve_path(
                self.train_path,
                "GD_TRAIN_PATH",
                "train.csv",
                r"D:\gradient_descent\train.csv",
            ),
            test_path=resolve_path(
                self.test_path,
                "GD_TEST_PATH",
                "test.csv",
                r"D:\gradient_descent\test.csv",
            ),
            sample_submission_path=resolve_path(
                self.sample_submission_path,
                "GD_SAMPLE_SUBMISSION_PATH",
                "sample_submission.csv",
                r"D:\gradient_descent\sample_submission.csv",
            ),
            target_column=self.target_column,
            id_columns=self.id_columns,
        )

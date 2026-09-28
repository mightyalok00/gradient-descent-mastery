from src.config import DataConfig


def test_data_config_preserves_target_and_ids():
    config = DataConfig(
        target_column="target",
        id_columns=("id",),
    )
    resolved = config.resolve()

    assert resolved.target_column == "target"
    assert resolved.id_columns == ("id",)

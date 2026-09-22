import numpy as np
import pandas as pd
import pytest

from pet_health_ai.livestock_data import LIVESTOCK_TASKS, generate_livestock_data, temperature_humidity_index


@pytest.fixture(scope="module")
def herd():
    return generate_livestock_data(hours=24, seed=17, missing_fraction=0, outlier_fraction=0)


def test_balanced_population_and_cadence(herd):
    assert len(herd) == 30 * 24 * 60
    assert herd.groupby("species").animal_id.nunique().to_dict() == {"buffalo": 15, "cow": 15}
    assert herd.groupby(["region_id", "species"]).animal_id.nunique().eq(3).all()
    assert herd.groupby("animal_id").region_id.nunique().eq(1).all()
    assert herd.animal_id.eq(herd.pet_id).all()
    assert not herd.duplicated(["animal_id", "timestamp"]).any()
    assert herd.groupby("animal_id").timestamp.diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_environment_and_target_semantics(herd):
    expected = temperature_humidity_index(herd.ambient_temperature_c, herd.relative_humidity_pct)
    assert np.allclose(herd.temperature_humidity_index, expected)
    assert herd.groupby(["region_id", "timestamp"]).ambient_temperature_c.nunique().eq(1).all()
    assert herd.relative_humidity_pct.between(0, 100).all()
    assert herd.loc[herd.region_id.eq("JK"), "ambient_temperature_c"].min() < 0
    assert herd.loc[herd.region_id.eq("RJ"), "ambient_temperature_c"].max() > 40
    targets = [f"target_{task}" for task in LIVESTOCK_TASKS]
    assert herd.label.eq(herd[targets].max(axis=1)).all()
    assert (herd[targets].sum() > 0).all()
    assert herd.loc[herd.target_heat_stress.eq(1), "target_fever"].eq(0).any()
    # Both species have injected examples of every supervised episode type.
    assert herd.groupby("species")[targets[:4]].sum().gt(0).all().all()
    assert np.isfinite(herd.select_dtypes(include="number")).all().all()
    assert herd.temperature.between(35, 42.5).all()


def test_reproducible_and_quality_flags():
    kwargs = dict(cows=5, buffaloes=5, hours=24, sample_minutes=5, seed=24,
                  missing_fraction=.01, outlier_fraction=.01)
    one = generate_livestock_data(**kwargs)
    two = generate_livestock_data(**kwargs)
    pd.testing.assert_frame_equal(one, two)
    sensors = ["heart_rate", "spo2", "temperature", "light"]
    assert one[sensors].isna().sum().sum() == one.sensor_missing_count.sum() > 0
    ranges = {"heart_rate": (30, 300), "spo2": (70, 100), "temperature": (35, 43), "light": (0, 200000)}
    bad = sum((one[c].notna() & ~one[c].between(*limits)).sum() for c, limits in ranges.items())
    assert bad == one.sensor_outlier_count.sum() > 0


@pytest.mark.parametrize("kwargs", [{"cows": 14}, {"buffaloes": 0}, {"hours": 0}, {"sample_minutes": 0}, {"missing_fraction": -.1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        generate_livestock_data(**kwargs)

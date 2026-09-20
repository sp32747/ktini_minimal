import numpy as np
from pet_health_ai.data_generation import SUPERVISED_TASKS, generate_synthetic_data
from pet_health_ai.features import clean_sensor_data, create_window_dataset


def test_cleaning_and_multitask_windows_are_finite():
    df=generate_synthetic_data(n_pets=5,hours_per_pet=4,seed=11); clean=clean_sensor_data(df)
    assert clean[["heart_rate","spo2","temperature","light","gps_lat","gps_lon"]].isna().sum().sum()==0
    windows=create_window_dataset(clean,window_minutes=15,stride_minutes=5)
    assert len(windows.features)==len(windows.labels)==len(windows.sequences)==len(windows.task_labels)
    assert windows.task_labels.shape[1]==len(SUPERVISED_TASKS); assert windows.sequences.shape[1]==15
    assert np.isfinite(windows.features.to_numpy()).all(); assert np.isfinite(windows.sequences).all()

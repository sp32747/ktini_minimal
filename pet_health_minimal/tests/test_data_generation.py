import numpy as np
from pet_health_ai.data_generation import SUPERVISED_TASKS, generate_synthetic_data


def test_synthetic_data_has_multitask_targets():
    df=generate_synthetic_data(n_pets=5,hours_per_pet=4,seed=7,missing_fraction=0,outlier_fraction=0)
    required={"pet_id","timestamp","heart_rate","spo2","temperature","light","gps_lat","gps_lon","label",*[f"target_{t}" for t in SUPERVISED_TASKS]}
    assert required.issubset(df.columns); assert df.pet_id.nunique()==5; assert set(np.unique(df.label)).issubset({0,1})
    for task in SUPERVISED_TASKS: assert df[f"target_{task}"].sum()>0

import pandas as pd
from pet_health_ai.health_score import add_longitudinal_health_score


def test_longitudinal_score_stays_bounded_and_is_per_pet():
    df=pd.DataFrame({
        "pet_id":["P1","P1","P2","P2"],"window_end":pd.date_range("2026-01-01",periods=4,freq="5min"),
        "hypoxemia_probability":[0.1,0.9,0.1,0.1],"fever_probability":[0.1]*4,
        "abnormal_resting_hr_probability":[0.1]*4,"activity_reduction_probability":[0.1]*4,"general_anomaly_probability":[0.1]*4,
    })
    out=add_longitudinal_health_score(df)
    assert out.health_score_0_100.between(0,100).all()
    p1=out[out.pet_id=="P1"].sort_values("window_end"); assert p1.health_score_0_100.iloc[1] < p1.health_score_0_100.iloc[0]

# Results of experiments

| $\rho$ type        | prob-estimator| inversion| prob-estimator | inversion| prob-estimator | inversion|proj-langevin (3q)| proj-langevin (4q)|
|    ----            |   ---      | - | -- |--| -- | -- |--|--|
|                    | python        | python   | R              | R        |paper | paper| matlab|matlab|
| rank-1             | **1.03e-5**   | 1.23e-5  | 3.11e-5        |**7.79e-6** | 5.13e-5|**1.55e-5**|6.96e-1|9.81e-1|
| rank-2             | **2.01e-5**   | 2.11e-5  | **5.43e-4**    | 1.5e-2   | **7.84e-3** | 1.58e-2|2.5e-3|2.5e-1|
| Approximate rank-2 | 1.88e-5       | 1.88e-5  | **7.51e-4**    | 1.58e-2  | **7.39e-3** | 1.52e-2|2.5e-3|2.42e-1|
| Maximal mixed state| 3.93e-4       |**3.14e-4**| **2.41e-4**   | 3.67e-4  | **3.20e-4** | 4.76e-4|1.22e-2|2.11e-1|

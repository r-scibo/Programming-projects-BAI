\* Bakery_Schedule *\
Minimize
Total_time_of_baking_schedule: end - start
Subject To
Bake_after_ready_0: s_s_0 >= 4800
Bake_after_ready_1: s_s_1 >= 5400
Bake_after_ready_10: s_s_10 >= 1800
Bake_after_ready_11: s_s_11 >= 15000
Bake_after_ready_12: s_s_12 >= 13800
Bake_after_ready_13: s_s_13 >= 2400
Bake_after_ready_14: s_s_14 >= 9600
Bake_after_ready_15: s_s_15 >= 3600
Bake_after_ready_16: s_s_16 >= 1800
Bake_after_ready_2: s_s_2 >= 0
Bake_after_ready_3: s_s_3 >= 3000
Bake_after_ready_4: s_s_4 >= 1200
Bake_after_ready_5: s_s_5 >= 3000
Bake_after_ready_6: s_s_6 >= 9000
Bake_after_ready_7: s_s_7 >= 8400
Bake_after_ready_8: s_s_8 >= 1200
Bake_after_ready_9: s_s_9 >= 600
Lower_bound_on_end_0: - end + s_s_0 <= -600
Lower_bound_on_end_1: - end + s_s_1 <= -600
Lower_bound_on_end_10: - end + s_s_10 <= -1200
Lower_bound_on_end_11: - end + s_s_11 <= -1800
Lower_bound_on_end_12: - end + s_s_12 <= -1800
Lower_bound_on_end_13: - end + s_s_13 <= -1800
Lower_bound_on_end_14: - end + s_s_14 <= -1800
Lower_bound_on_end_15: - end + s_s_15 <= -2400
Lower_bound_on_end_16: - end + s_s_16 <= -2400
Lower_bound_on_end_2: - end + s_s_2 <= -600
Lower_bound_on_end_3: - end + s_s_3 <= -600
Lower_bound_on_end_4: - end + s_s_4 <= -600
Lower_bound_on_end_5: - end + s_s_5 <= -1200
Lower_bound_on_end_6: - end + s_s_6 <= -1200
Lower_bound_on_end_7: - end + s_s_7 <= -1200
Lower_bound_on_end_8: - end + s_s_8 <= -1200
Lower_bound_on_end_9: - end + s_s_9 <= -1200
Non_overlapping_0_10_constraint_1: s_s_0 - s_s_10 + 27000 u_0_10 <= 26400
Non_overlapping_0_10_constraint_2: - s_s_0 + s_s_10 - 27000 u_0_10 <= -1200
Non_overlapping_0_11_constraint_1: s_s_0 - s_s_11 + 27000 u_0_11 <= 26400
Non_overlapping_0_11_constraint_2: - s_s_0 + s_s_11 - 27000 u_0_11 <= -1800
Non_overlapping_0_12_constraint_1: s_s_0 - s_s_12 + 27000 u_0_12 <= 26400
Non_overlapping_0_12_constraint_2: - s_s_0 + s_s_12 - 27000 u_0_12 <= -1800
Non_overlapping_0_13_constraint_1: s_s_0 - s_s_13 + 27000 u_0_13 <= 26400
Non_overlapping_0_13_constraint_2: - s_s_0 + s_s_13 - 27000 u_0_13 <= -1800
Non_overlapping_0_14_constraint_1: s_s_0 - s_s_14 + 27000 u_0_14 <= 26400
Non_overlapping_0_14_constraint_2: - s_s_0 + s_s_14 - 27000 u_0_14 <= -1800
Non_overlapping_0_15_constraint_1: s_s_0 - s_s_15 + 27000 u_0_15 <= 26400
Non_overlapping_0_15_constraint_2: - s_s_0 + s_s_15 - 27000 u_0_15 <= -2400
Non_overlapping_0_16_constraint_1: s_s_0 - s_s_16 + 27000 u_0_16 <= 26400
Non_overlapping_0_16_constraint_2: - s_s_0 + s_s_16 - 27000 u_0_16 <= -2400
Non_overlapping_0_1_constraint_1: s_s_0 - s_s_1 + 27000 u_0_1 <= 26400
Non_overlapping_0_1_constraint_2: - s_s_0 + s_s_1 - 27000 u_0_1 <= -600
Non_overlapping_0_2_constraint_1: s_s_0 - s_s_2 + 27000 u_0_2 <= 26400
Non_overlapping_0_2_constraint_2: - s_s_0 + s_s_2 - 27000 u_0_2 <= -600
Non_overlapping_0_3_constraint_1: s_s_0 - s_s_3 + 27000 u_0_3 <= 26400
Non_overlapping_0_3_constraint_2: - s_s_0 + s_s_3 - 27000 u_0_3 <= -600
Non_overlapping_0_4_constraint_1: s_s_0 - s_s_4 + 27000 u_0_4 <= 26400
Non_overlapping_0_4_constraint_2: - s_s_0 + s_s_4 - 27000 u_0_4 <= -600
Non_overlapping_0_5_constraint_1: s_s_0 - s_s_5 + 27000 u_0_5 <= 26400
Non_overlapping_0_5_constraint_2: - s_s_0 + s_s_5 - 27000 u_0_5 <= -1200
Non_overlapping_0_6_constraint_1: s_s_0 - s_s_6 + 27000 u_0_6 <= 26400
Non_overlapping_0_6_constraint_2: - s_s_0 + s_s_6 - 27000 u_0_6 <= -1200
Non_overlapping_0_7_constraint_1: s_s_0 - s_s_7 + 27000 u_0_7 <= 26400
Non_overlapping_0_7_constraint_2: - s_s_0 + s_s_7 - 27000 u_0_7 <= -1200
Non_overlapping_0_8_constraint_1: s_s_0 - s_s_8 + 27000 u_0_8 <= 26400
Non_overlapping_0_8_constraint_2: - s_s_0 + s_s_8 - 27000 u_0_8 <= -1200
Non_overlapping_0_9_constraint_1: s_s_0 - s_s_9 + 27000 u_0_9 <= 26400
Non_overlapping_0_9_constraint_2: - s_s_0 + s_s_9 - 27000 u_0_9 <= -1200
Non_overlapping_10_0_constraint_1: - s_s_0 + s_s_10 + 27000 u_10_0 <= 25800
Non_overlapping_10_0_constraint_2: s_s_0 - s_s_10 - 27000 u_10_0 <= -600
Non_overlapping_10_11_constraint_1: s_s_10 - s_s_11 + 27000 u_10_11 <= 25800
Non_overlapping_10_11_constraint_2: - s_s_10 + s_s_11 - 27000 u_10_11 <= -1800
Non_overlapping_10_12_constraint_1: s_s_10 - s_s_12 + 27000 u_10_12 <= 25800
Non_overlapping_10_12_constraint_2: - s_s_10 + s_s_12 - 27000 u_10_12 <= -1800
Non_overlapping_10_13_constraint_1: s_s_10 - s_s_13 + 27000 u_10_13 <= 25800
Non_overlapping_10_13_constraint_2: - s_s_10 + s_s_13 - 27000 u_10_13 <= -1800
Non_overlapping_10_14_constraint_1: s_s_10 - s_s_14 + 27000 u_10_14 <= 25800
Non_overlapping_10_14_constraint_2: - s_s_10 + s_s_14 - 27000 u_10_14 <= -1800
Non_overlapping_10_15_constraint_1: s_s_10 - s_s_15 + 27000 u_10_15 <= 25800
Non_overlapping_10_15_constraint_2: - s_s_10 + s_s_15 - 27000 u_10_15 <= -2400
Non_overlapping_10_16_constraint_1: s_s_10 - s_s_16 + 27000 u_10_16 <= 25800
Non_overlapping_10_16_constraint_2: - s_s_10 + s_s_16 - 27000 u_10_16 <= -2400
Non_overlapping_10_1_constraint_1: - s_s_1 + s_s_10 + 27000 u_10_1 <= 25800
Non_overlapping_10_1_constraint_2: s_s_1 - s_s_10 - 27000 u_10_1 <= -600
Non_overlapping_10_2_constraint_1: s_s_10 - s_s_2 + 27000 u_10_2 <= 25800
Non_overlapping_10_2_constraint_2: - s_s_10 + s_s_2 - 27000 u_10_2 <= -600
Non_overlapping_10_3_constraint_1: s_s_10 - s_s_3 + 27000 u_10_3 <= 25800
Non_overlapping_10_3_constraint_2: - s_s_10 + s_s_3 - 27000 u_10_3 <= -600
Non_overlapping_10_4_constraint_1: s_s_10 - s_s_4 + 27000 u_10_4 <= 25800
Non_overlapping_10_4_constraint_2: - s_s_10 + s_s_4 - 27000 u_10_4 <= -600
Non_overlapping_10_5_constraint_1: s_s_10 - s_s_5 + 27000 u_10_5 <= 25800
Non_overlapping_10_5_constraint_2: - s_s_10 + s_s_5 - 27000 u_10_5 <= -1200
Non_overlapping_10_6_constraint_1: s_s_10 - s_s_6 + 27000 u_10_6 <= 25800
Non_overlapping_10_6_constraint_2: - s_s_10 + s_s_6 - 27000 u_10_6 <= -1200
Non_overlapping_10_7_constraint_1: s_s_10 - s_s_7 + 27000 u_10_7 <= 25800
Non_overlapping_10_7_constraint_2: - s_s_10 + s_s_7 - 27000 u_10_7 <= -1200
Non_overlapping_10_8_constraint_1: s_s_10 - s_s_8 + 27000 u_10_8 <= 25800
Non_overlapping_10_8_constraint_2: - s_s_10 + s_s_8 - 27000 u_10_8 <= -1200
Non_overlapping_10_9_constraint_1: s_s_10 - s_s_9 + 27000 u_10_9 <= 25800
Non_overlapping_10_9_constraint_2: - s_s_10 + s_s_9 - 27000 u_10_9 <= -1200
Non_overlapping_11_0_constraint_1: - s_s_0 + s_s_11 + 27000 u_11_0 <= 25200
Non_overlapping_11_0_constraint_2: s_s_0 - s_s_11 - 27000 u_11_0 <= -600
Non_overlapping_11_10_constraint_1: - s_s_10 + s_s_11 + 27000 u_11_10 <= 25200
Non_overlapping_11_10_constraint_2: s_s_10 - s_s_11 - 27000 u_11_10 <= -1200
Non_overlapping_11_12_constraint_1: s_s_11 - s_s_12 + 27000 u_11_12 <= 25200
Non_overlapping_11_12_constraint_2: - s_s_11 + s_s_12 - 27000 u_11_12 <= -1800
Non_overlapping_11_13_constraint_1: s_s_11 - s_s_13 + 27000 u_11_13 <= 25200
Non_overlapping_11_13_constraint_2: - s_s_11 + s_s_13 - 27000 u_11_13 <= -1800
Non_overlapping_11_14_constraint_1: s_s_11 - s_s_14 + 27000 u_11_14 <= 25200
Non_overlapping_11_14_constraint_2: - s_s_11 + s_s_14 - 27000 u_11_14 <= -1800
Non_overlapping_11_15_constraint_1: s_s_11 - s_s_15 + 27000 u_11_15 <= 25200
Non_overlapping_11_15_constraint_2: - s_s_11 + s_s_15 - 27000 u_11_15 <= -2400
Non_overlapping_11_16_constraint_1: s_s_11 - s_s_16 + 27000 u_11_16 <= 25200
Non_overlapping_11_16_constraint_2: - s_s_11 + s_s_16 - 27000 u_11_16 <= -2400
Non_overlapping_11_1_constraint_1: - s_s_1 + s_s_11 + 27000 u_11_1 <= 25200
Non_overlapping_11_1_constraint_2: s_s_1 - s_s_11 - 27000 u_11_1 <= -600
Non_overlapping_11_2_constraint_1: s_s_11 - s_s_2 + 27000 u_11_2 <= 25200
Non_overlapping_11_2_constraint_2: - s_s_11 + s_s_2 - 27000 u_11_2 <= -600
Non_overlapping_11_3_constraint_1: s_s_11 - s_s_3 + 27000 u_11_3 <= 25200
Non_overlapping_11_3_constraint_2: - s_s_11 + s_s_3 - 27000 u_11_3 <= -600
Non_overlapping_11_4_constraint_1: s_s_11 - s_s_4 + 27000 u_11_4 <= 25200
Non_overlapping_11_4_constraint_2: - s_s_11 + s_s_4 - 27000 u_11_4 <= -600
Non_overlapping_11_5_constraint_1: s_s_11 - s_s_5 + 27000 u_11_5 <= 25200
Non_overlapping_11_5_constraint_2: - s_s_11 + s_s_5 - 27000 u_11_5 <= -1200
Non_overlapping_11_6_constraint_1: s_s_11 - s_s_6 + 27000 u_11_6 <= 25200
Non_overlapping_11_6_constraint_2: - s_s_11 + s_s_6 - 27000 u_11_6 <= -1200
Non_overlapping_11_7_constraint_1: s_s_11 - s_s_7 + 27000 u_11_7 <= 25200
Non_overlapping_11_7_constraint_2: - s_s_11 + s_s_7 - 27000 u_11_7 <= -1200
Non_overlapping_11_8_constraint_1: s_s_11 - s_s_8 + 27000 u_11_8 <= 25200
Non_overlapping_11_8_constraint_2: - s_s_11 + s_s_8 - 27000 u_11_8 <= -1200
Non_overlapping_11_9_constraint_1: s_s_11 - s_s_9 + 27000 u_11_9 <= 25200
Non_overlapping_11_9_constraint_2: - s_s_11 + s_s_9 - 27000 u_11_9 <= -1200
Non_overlapping_12_0_constraint_1: - s_s_0 + s_s_12 + 27000 u_12_0 <= 25200
Non_overlapping_12_0_constraint_2: s_s_0 - s_s_12 - 27000 u_12_0 <= -600
Non_overlapping_12_10_constraint_1: - s_s_10 + s_s_12 + 27000 u_12_10 <= 25200
Non_overlapping_12_10_constraint_2: s_s_10 - s_s_12 - 27000 u_12_10 <= -1200
Non_overlapping_12_11_constraint_1: - s_s_11 + s_s_12 + 27000 u_12_11 <= 25200
Non_overlapping_12_11_constraint_2: s_s_11 - s_s_12 - 27000 u_12_11 <= -1800
Non_overlapping_12_13_constraint_1: s_s_12 - s_s_13 + 27000 u_12_13 <= 25200
Non_overlapping_12_13_constraint_2: - s_s_12 + s_s_13 - 27000 u_12_13 <= -1800
Non_overlapping_12_14_constraint_1: s_s_12 - s_s_14 + 27000 u_12_14 <= 25200
Non_overlapping_12_14_constraint_2: - s_s_12 + s_s_14 - 27000 u_12_14 <= -1800
Non_overlapping_12_15_constraint_1: s_s_12 - s_s_15 + 27000 u_12_15 <= 25200
Non_overlapping_12_15_constraint_2: - s_s_12 + s_s_15 - 27000 u_12_15 <= -2400
Non_overlapping_12_16_constraint_1: s_s_12 - s_s_16 + 27000 u_12_16 <= 25200
Non_overlapping_12_16_constraint_2: - s_s_12 + s_s_16 - 27000 u_12_16 <= -2400
Non_overlapping_12_1_constraint_1: - s_s_1 + s_s_12 + 27000 u_12_1 <= 25200
Non_overlapping_12_1_constraint_2: s_s_1 - s_s_12 - 27000 u_12_1 <= -600
Non_overlapping_12_2_constraint_1: s_s_12 - s_s_2 + 27000 u_12_2 <= 25200
Non_overlapping_12_2_constraint_2: - s_s_12 + s_s_2 - 27000 u_12_2 <= -600
Non_overlapping_12_3_constraint_1: s_s_12 - s_s_3 + 27000 u_12_3 <= 25200
Non_overlapping_12_3_constraint_2: - s_s_12 + s_s_3 - 27000 u_12_3 <= -600
Non_overlapping_12_4_constraint_1: s_s_12 - s_s_4 + 27000 u_12_4 <= 25200
Non_overlapping_12_4_constraint_2: - s_s_12 + s_s_4 - 27000 u_12_4 <= -600
Non_overlapping_12_5_constraint_1: s_s_12 - s_s_5 + 27000 u_12_5 <= 25200
Non_overlapping_12_5_constraint_2: - s_s_12 + s_s_5 - 27000 u_12_5 <= -1200
Non_overlapping_12_6_constraint_1: s_s_12 - s_s_6 + 27000 u_12_6 <= 25200
Non_overlapping_12_6_constraint_2: - s_s_12 + s_s_6 - 27000 u_12_6 <= -1200
Non_overlapping_12_7_constraint_1: s_s_12 - s_s_7 + 27000 u_12_7 <= 25200
Non_overlapping_12_7_constraint_2: - s_s_12 + s_s_7 - 27000 u_12_7 <= -1200
Non_overlapping_12_8_constraint_1: s_s_12 - s_s_8 + 27000 u_12_8 <= 25200
Non_overlapping_12_8_constraint_2: - s_s_12 + s_s_8 - 27000 u_12_8 <= -1200
Non_overlapping_12_9_constraint_1: s_s_12 - s_s_9 + 27000 u_12_9 <= 25200
Non_overlapping_12_9_constraint_2: - s_s_12 + s_s_9 - 27000 u_12_9 <= -1200
Non_overlapping_13_0_constraint_1: - s_s_0 + s_s_13 + 27000 u_13_0 <= 25200
Non_overlapping_13_0_constraint_2: s_s_0 - s_s_13 - 27000 u_13_0 <= -600
Non_overlapping_13_10_constraint_1: - s_s_10 + s_s_13 + 27000 u_13_10 <= 25200
Non_overlapping_13_10_constraint_2: s_s_10 - s_s_13 - 27000 u_13_10 <= -1200
Non_overlapping_13_11_constraint_1: - s_s_11 + s_s_13 + 27000 u_13_11 <= 25200
Non_overlapping_13_11_constraint_2: s_s_11 - s_s_13 - 27000 u_13_11 <= -1800
Non_overlapping_13_12_constraint_1: - s_s_12 + s_s_13 + 27000 u_13_12 <= 25200
Non_overlapping_13_12_constraint_2: s_s_12 - s_s_13 - 27000 u_13_12 <= -1800
Non_overlapping_13_14_constraint_1: s_s_13 - s_s_14 + 27000 u_13_14 <= 25200
Non_overlapping_13_14_constraint_2: - s_s_13 + s_s_14 - 27000 u_13_14 <= -1800
Non_overlapping_13_15_constraint_1: s_s_13 - s_s_15 + 27000 u_13_15 <= 25200
Non_overlapping_13_15_constraint_2: - s_s_13 + s_s_15 - 27000 u_13_15 <= -2400
Non_overlapping_13_16_constraint_1: s_s_13 - s_s_16 + 27000 u_13_16 <= 25200
Non_overlapping_13_16_constraint_2: - s_s_13 + s_s_16 - 27000 u_13_16 <= -2400
Non_overlapping_13_1_constraint_1: - s_s_1 + s_s_13 + 27000 u_13_1 <= 25200
Non_overlapping_13_1_constraint_2: s_s_1 - s_s_13 - 27000 u_13_1 <= -600
Non_overlapping_13_2_constraint_1: s_s_13 - s_s_2 + 27000 u_13_2 <= 25200
Non_overlapping_13_2_constraint_2: - s_s_13 + s_s_2 - 27000 u_13_2 <= -600
Non_overlapping_13_3_constraint_1: s_s_13 - s_s_3 + 27000 u_13_3 <= 25200
Non_overlapping_13_3_constraint_2: - s_s_13 + s_s_3 - 27000 u_13_3 <= -600
Non_overlapping_13_4_constraint_1: s_s_13 - s_s_4 + 27000 u_13_4 <= 25200
Non_overlapping_13_4_constraint_2: - s_s_13 + s_s_4 - 27000 u_13_4 <= -600
Non_overlapping_13_5_constraint_1: s_s_13 - s_s_5 + 27000 u_13_5 <= 25200
Non_overlapping_13_5_constraint_2: - s_s_13 + s_s_5 - 27000 u_13_5 <= -1200
Non_overlapping_13_6_constraint_1: s_s_13 - s_s_6 + 27000 u_13_6 <= 25200
Non_overlapping_13_6_constraint_2: - s_s_13 + s_s_6 - 27000 u_13_6 <= -1200
Non_overlapping_13_7_constraint_1: s_s_13 - s_s_7 + 27000 u_13_7 <= 25200
Non_overlapping_13_7_constraint_2: - s_s_13 + s_s_7 - 27000 u_13_7 <= -1200
Non_overlapping_13_8_constraint_1: s_s_13 - s_s_8 + 27000 u_13_8 <= 25200
Non_overlapping_13_8_constraint_2: - s_s_13 + s_s_8 - 27000 u_13_8 <= -1200
Non_overlapping_13_9_constraint_1: s_s_13 - s_s_9 + 27000 u_13_9 <= 25200
Non_overlapping_13_9_constraint_2: - s_s_13 + s_s_9 - 27000 u_13_9 <= -1200
Non_overlapping_14_0_constraint_1: - s_s_0 + s_s_14 + 27000 u_14_0 <= 25200
Non_overlapping_14_0_constraint_2: s_s_0 - s_s_14 - 27000 u_14_0 <= -600
Non_overlapping_14_10_constraint_1: - s_s_10 + s_s_14 + 27000 u_14_10 <= 25200
Non_overlapping_14_10_constraint_2: s_s_10 - s_s_14 - 27000 u_14_10 <= -1200
Non_overlapping_14_11_constraint_1: - s_s_11 + s_s_14 + 27000 u_14_11 <= 25200
Non_overlapping_14_11_constraint_2: s_s_11 - s_s_14 - 27000 u_14_11 <= -1800
Non_overlapping_14_12_constraint_1: - s_s_12 + s_s_14 + 27000 u_14_12 <= 25200
Non_overlapping_14_12_constraint_2: s_s_12 - s_s_14 - 27000 u_14_12 <= -1800
Non_overlapping_14_13_constraint_1: - s_s_13 + s_s_14 + 27000 u_14_13 <= 25200
Non_overlapping_14_13_constraint_2: s_s_13 - s_s_14 - 27000 u_14_13 <= -1800
Non_overlapping_14_15_constraint_1: s_s_14 - s_s_15 + 27000 u_14_15 <= 25200
Non_overlapping_14_15_constraint_2: - s_s_14 + s_s_15 - 27000 u_14_15 <= -2400
Non_overlapping_14_16_constraint_1: s_s_14 - s_s_16 + 27000 u_14_16 <= 25200
Non_overlapping_14_16_constraint_2: - s_s_14 + s_s_16 - 27000 u_14_16 <= -2400
Non_overlapping_14_1_constraint_1: - s_s_1 + s_s_14 + 27000 u_14_1 <= 25200
Non_overlapping_14_1_constraint_2: s_s_1 - s_s_14 - 27000 u_14_1 <= -600
Non_overlapping_14_2_constraint_1: s_s_14 - s_s_2 + 27000 u_14_2 <= 25200
Non_overlapping_14_2_constraint_2: - s_s_14 + s_s_2 - 27000 u_14_2 <= -600
Non_overlapping_14_3_constraint_1: s_s_14 - s_s_3 + 27000 u_14_3 <= 25200
Non_overlapping_14_3_constraint_2: - s_s_14 + s_s_3 - 27000 u_14_3 <= -600
Non_overlapping_14_4_constraint_1: s_s_14 - s_s_4 + 27000 u_14_4 <= 25200
Non_overlapping_14_4_constraint_2: - s_s_14 + s_s_4 - 27000 u_14_4 <= -600
Non_overlapping_14_5_constraint_1: s_s_14 - s_s_5 + 27000 u_14_5 <= 25200
Non_overlapping_14_5_constraint_2: - s_s_14 + s_s_5 - 27000 u_14_5 <= -1200
Non_overlapping_14_6_constraint_1: s_s_14 - s_s_6 + 27000 u_14_6 <= 25200
Non_overlapping_14_6_constraint_2: - s_s_14 + s_s_6 - 27000 u_14_6 <= -1200
Non_overlapping_14_7_constraint_1: s_s_14 - s_s_7 + 27000 u_14_7 <= 25200
Non_overlapping_14_7_constraint_2: - s_s_14 + s_s_7 - 27000 u_14_7 <= -1200
Non_overlapping_14_8_constraint_1: s_s_14 - s_s_8 + 27000 u_14_8 <= 25200
Non_overlapping_14_8_constraint_2: - s_s_14 + s_s_8 - 27000 u_14_8 <= -1200
Non_overlapping_14_9_constraint_1: s_s_14 - s_s_9 + 27000 u_14_9 <= 25200
Non_overlapping_14_9_constraint_2: - s_s_14 + s_s_9 - 27000 u_14_9 <= -1200
Non_overlapping_15_0_constraint_1: - s_s_0 + s_s_15 + 27000 u_15_0 <= 24600
Non_overlapping_15_0_constraint_2: s_s_0 - s_s_15 - 27000 u_15_0 <= -600
Non_overlapping_15_10_constraint_1: - s_s_10 + s_s_15 + 27000 u_15_10 <= 24600
Non_overlapping_15_10_constraint_2: s_s_10 - s_s_15 - 27000 u_15_10 <= -1200
Non_overlapping_15_11_constraint_1: - s_s_11 + s_s_15 + 27000 u_15_11 <= 24600
Non_overlapping_15_11_constraint_2: s_s_11 - s_s_15 - 27000 u_15_11 <= -1800
Non_overlapping_15_12_constraint_1: - s_s_12 + s_s_15 + 27000 u_15_12 <= 24600
Non_overlapping_15_12_constraint_2: s_s_12 - s_s_15 - 27000 u_15_12 <= -1800
Non_overlapping_15_13_constraint_1: - s_s_13 + s_s_15 + 27000 u_15_13 <= 24600
Non_overlapping_15_13_constraint_2: s_s_13 - s_s_15 - 27000 u_15_13 <= -1800
Non_overlapping_15_14_constraint_1: - s_s_14 + s_s_15 + 27000 u_15_14 <= 24600
Non_overlapping_15_14_constraint_2: s_s_14 - s_s_15 - 27000 u_15_14 <= -1800
Non_overlapping_15_16_constraint_1: s_s_15 - s_s_16 + 27000 u_15_16 <= 24600
Non_overlapping_15_16_constraint_2: - s_s_15 + s_s_16 - 27000 u_15_16 <= -2400
Non_overlapping_15_1_constraint_1: - s_s_1 + s_s_15 + 27000 u_15_1 <= 24600
Non_overlapping_15_1_constraint_2: s_s_1 - s_s_15 - 27000 u_15_1 <= -600
Non_overlapping_15_2_constraint_1: s_s_15 - s_s_2 + 27000 u_15_2 <= 24600
Non_overlapping_15_2_constraint_2: - s_s_15 + s_s_2 - 27000 u_15_2 <= -600
Non_overlapping_15_3_constraint_1: s_s_15 - s_s_3 + 27000 u_15_3 <= 24600
Non_overlapping_15_3_constraint_2: - s_s_15 + s_s_3 - 27000 u_15_3 <= -600
Non_overlapping_15_4_constraint_1: s_s_15 - s_s_4 + 27000 u_15_4 <= 24600
Non_overlapping_15_4_constraint_2: - s_s_15 + s_s_4 - 27000 u_15_4 <= -600
Non_overlapping_15_5_constraint_1: s_s_15 - s_s_5 + 27000 u_15_5 <= 24600
Non_overlapping_15_5_constraint_2: - s_s_15 + s_s_5 - 27000 u_15_5 <= -1200
Non_overlapping_15_6_constraint_1: s_s_15 - s_s_6 + 27000 u_15_6 <= 24600
Non_overlapping_15_6_constraint_2: - s_s_15 + s_s_6 - 27000 u_15_6 <= -1200
Non_overlapping_15_7_constraint_1: s_s_15 - s_s_7 + 27000 u_15_7 <= 24600
Non_overlapping_15_7_constraint_2: - s_s_15 + s_s_7 - 27000 u_15_7 <= -1200
Non_overlapping_15_8_constraint_1: s_s_15 - s_s_8 + 27000 u_15_8 <= 24600
Non_overlapping_15_8_constraint_2: - s_s_15 + s_s_8 - 27000 u_15_8 <= -1200
Non_overlapping_15_9_constraint_1: s_s_15 - s_s_9 + 27000 u_15_9 <= 24600
Non_overlapping_15_9_constraint_2: - s_s_15 + s_s_9 - 27000 u_15_9 <= -1200
Non_overlapping_16_0_constraint_1: - s_s_0 + s_s_16 + 27000 u_16_0 <= 24600
Non_overlapping_16_0_constraint_2: s_s_0 - s_s_16 - 27000 u_16_0 <= -600
Non_overlapping_16_10_constraint_1: - s_s_10 + s_s_16 + 27000 u_16_10 <= 24600
Non_overlapping_16_10_constraint_2: s_s_10 - s_s_16 - 27000 u_16_10 <= -1200
Non_overlapping_16_11_constraint_1: - s_s_11 + s_s_16 + 27000 u_16_11 <= 24600
Non_overlapping_16_11_constraint_2: s_s_11 - s_s_16 - 27000 u_16_11 <= -1800
Non_overlapping_16_12_constraint_1: - s_s_12 + s_s_16 + 27000 u_16_12 <= 24600
Non_overlapping_16_12_constraint_2: s_s_12 - s_s_16 - 27000 u_16_12 <= -1800
Non_overlapping_16_13_constraint_1: - s_s_13 + s_s_16 + 27000 u_16_13 <= 24600
Non_overlapping_16_13_constraint_2: s_s_13 - s_s_16 - 27000 u_16_13 <= -1800
Non_overlapping_16_14_constraint_1: - s_s_14 + s_s_16 + 27000 u_16_14 <= 24600
Non_overlapping_16_14_constraint_2: s_s_14 - s_s_16 - 27000 u_16_14 <= -1800
Non_overlapping_16_15_constraint_1: - s_s_15 + s_s_16 + 27000 u_16_15 <= 24600
Non_overlapping_16_15_constraint_2: s_s_15 - s_s_16 - 27000 u_16_15 <= -2400
Non_overlapping_16_1_constraint_1: - s_s_1 + s_s_16 + 27000 u_16_1 <= 24600
Non_overlapping_16_1_constraint_2: s_s_1 - s_s_16 - 27000 u_16_1 <= -600
Non_overlapping_16_2_constraint_1: s_s_16 - s_s_2 + 27000 u_16_2 <= 24600
Non_overlapping_16_2_constraint_2: - s_s_16 + s_s_2 - 27000 u_16_2 <= -600
Non_overlapping_16_3_constraint_1: s_s_16 - s_s_3 + 27000 u_16_3 <= 24600
Non_overlapping_16_3_constraint_2: - s_s_16 + s_s_3 - 27000 u_16_3 <= -600
Non_overlapping_16_4_constraint_1: s_s_16 - s_s_4 + 27000 u_16_4 <= 24600
Non_overlapping_16_4_constraint_2: - s_s_16 + s_s_4 - 27000 u_16_4 <= -600
Non_overlapping_16_5_constraint_1: s_s_16 - s_s_5 + 27000 u_16_5 <= 24600
Non_overlapping_16_5_constraint_2: - s_s_16 + s_s_5 - 27000 u_16_5 <= -1200
Non_overlapping_16_6_constraint_1: s_s_16 - s_s_6 + 27000 u_16_6 <= 24600
Non_overlapping_16_6_constraint_2: - s_s_16 + s_s_6 - 27000 u_16_6 <= -1200
Non_overlapping_16_7_constraint_1: s_s_16 - s_s_7 + 27000 u_16_7 <= 24600
Non_overlapping_16_7_constraint_2: - s_s_16 + s_s_7 - 27000 u_16_7 <= -1200
Non_overlapping_16_8_constraint_1: s_s_16 - s_s_8 + 27000 u_16_8 <= 24600
Non_overlapping_16_8_constraint_2: - s_s_16 + s_s_8 - 27000 u_16_8 <= -1200
Non_overlapping_16_9_constraint_1: s_s_16 - s_s_9 + 27000 u_16_9 <= 24600
Non_overlapping_16_9_constraint_2: - s_s_16 + s_s_9 - 27000 u_16_9 <= -1200
Non_overlapping_1_0_constraint_1: - s_s_0 + s_s_1 + 27000 u_1_0 <= 26400
Non_overlapping_1_0_constraint_2: s_s_0 - s_s_1 - 27000 u_1_0 <= -600
Non_overlapping_1_10_constraint_1: s_s_1 - s_s_10 + 27000 u_1_10 <= 26400
Non_overlapping_1_10_constraint_2: - s_s_1 + s_s_10 - 27000 u_1_10 <= -1200
Non_overlapping_1_11_constraint_1: s_s_1 - s_s_11 + 27000 u_1_11 <= 26400
Non_overlapping_1_11_constraint_2: - s_s_1 + s_s_11 - 27000 u_1_11 <= -1800
Non_overlapping_1_12_constraint_1: s_s_1 - s_s_12 + 27000 u_1_12 <= 26400
Non_overlapping_1_12_constraint_2: - s_s_1 + s_s_12 - 27000 u_1_12 <= -1800
Non_overlapping_1_13_constraint_1: s_s_1 - s_s_13 + 27000 u_1_13 <= 26400
Non_overlapping_1_13_constraint_2: - s_s_1 + s_s_13 - 27000 u_1_13 <= -1800
Non_overlapping_1_14_constraint_1: s_s_1 - s_s_14 + 27000 u_1_14 <= 26400
Non_overlapping_1_14_constraint_2: - s_s_1 + s_s_14 - 27000 u_1_14 <= -1800
Non_overlapping_1_15_constraint_1: s_s_1 - s_s_15 + 27000 u_1_15 <= 26400
Non_overlapping_1_15_constraint_2: - s_s_1 + s_s_15 - 27000 u_1_15 <= -2400
Non_overlapping_1_16_constraint_1: s_s_1 - s_s_16 + 27000 u_1_16 <= 26400
Non_overlapping_1_16_constraint_2: - s_s_1 + s_s_16 - 27000 u_1_16 <= -2400
Non_overlapping_1_2_constraint_1: s_s_1 - s_s_2 + 27000 u_1_2 <= 26400
Non_overlapping_1_2_constraint_2: - s_s_1 + s_s_2 - 27000 u_1_2 <= -600
Non_overlapping_1_3_constraint_1: s_s_1 - s_s_3 + 27000 u_1_3 <= 26400
Non_overlapping_1_3_constraint_2: - s_s_1 + s_s_3 - 27000 u_1_3 <= -600
Non_overlapping_1_4_constraint_1: s_s_1 - s_s_4 + 27000 u_1_4 <= 26400
Non_overlapping_1_4_constraint_2: - s_s_1 + s_s_4 - 27000 u_1_4 <= -600
Non_overlapping_1_5_constraint_1: s_s_1 - s_s_5 + 27000 u_1_5 <= 26400
Non_overlapping_1_5_constraint_2: - s_s_1 + s_s_5 - 27000 u_1_5 <= -1200
Non_overlapping_1_6_constraint_1: s_s_1 - s_s_6 + 27000 u_1_6 <= 26400
Non_overlapping_1_6_constraint_2: - s_s_1 + s_s_6 - 27000 u_1_6 <= -1200
Non_overlapping_1_7_constraint_1: s_s_1 - s_s_7 + 27000 u_1_7 <= 26400
Non_overlapping_1_7_constraint_2: - s_s_1 + s_s_7 - 27000 u_1_7 <= -1200
Non_overlapping_1_8_constraint_1: s_s_1 - s_s_8 + 27000 u_1_8 <= 26400
Non_overlapping_1_8_constraint_2: - s_s_1 + s_s_8 - 27000 u_1_8 <= -1200
Non_overlapping_1_9_constraint_1: s_s_1 - s_s_9 + 27000 u_1_9 <= 26400
Non_overlapping_1_9_constraint_2: - s_s_1 + s_s_9 - 27000 u_1_9 <= -1200
Non_overlapping_2_0_constraint_1: - s_s_0 + s_s_2 + 27000 u_2_0 <= 26400
Non_overlapping_2_0_constraint_2: s_s_0 - s_s_2 - 27000 u_2_0 <= -600
Non_overlapping_2_10_constraint_1: - s_s_10 + s_s_2 + 27000 u_2_10 <= 26400
Non_overlapping_2_10_constraint_2: s_s_10 - s_s_2 - 27000 u_2_10 <= -1200
Non_overlapping_2_11_constraint_1: - s_s_11 + s_s_2 + 27000 u_2_11 <= 26400
Non_overlapping_2_11_constraint_2: s_s_11 - s_s_2 - 27000 u_2_11 <= -1800
Non_overlapping_2_12_constraint_1: - s_s_12 + s_s_2 + 27000 u_2_12 <= 26400
Non_overlapping_2_12_constraint_2: s_s_12 - s_s_2 - 27000 u_2_12 <= -1800
Non_overlapping_2_13_constraint_1: - s_s_13 + s_s_2 + 27000 u_2_13 <= 26400
Non_overlapping_2_13_constraint_2: s_s_13 - s_s_2 - 27000 u_2_13 <= -1800
Non_overlapping_2_14_constraint_1: - s_s_14 + s_s_2 + 27000 u_2_14 <= 26400
Non_overlapping_2_14_constraint_2: s_s_14 - s_s_2 - 27000 u_2_14 <= -1800
Non_overlapping_2_15_constraint_1: - s_s_15 + s_s_2 + 27000 u_2_15 <= 26400
Non_overlapping_2_15_constraint_2: s_s_15 - s_s_2 - 27000 u_2_15 <= -2400
Non_overlapping_2_16_constraint_1: - s_s_16 + s_s_2 + 27000 u_2_16 <= 26400
Non_overlapping_2_16_constraint_2: s_s_16 - s_s_2 - 27000 u_2_16 <= -2400
Non_overlapping_2_1_constraint_1: - s_s_1 + s_s_2 + 27000 u_2_1 <= 26400
Non_overlapping_2_1_constraint_2: s_s_1 - s_s_2 - 27000 u_2_1 <= -600
Non_overlapping_2_3_constraint_1: s_s_2 - s_s_3 + 27000 u_2_3 <= 26400
Non_overlapping_2_3_constraint_2: - s_s_2 + s_s_3 - 27000 u_2_3 <= -600
Non_overlapping_2_4_constraint_1: s_s_2 - s_s_4 + 27000 u_2_4 <= 26400
Non_overlapping_2_4_constraint_2: - s_s_2 + s_s_4 - 27000 u_2_4 <= -600
Non_overlapping_2_5_constraint_1: s_s_2 - s_s_5 + 27000 u_2_5 <= 26400
Non_overlapping_2_5_constraint_2: - s_s_2 + s_s_5 - 27000 u_2_5 <= -1200
Non_overlapping_2_6_constraint_1: s_s_2 - s_s_6 + 27000 u_2_6 <= 26400
Non_overlapping_2_6_constraint_2: - s_s_2 + s_s_6 - 27000 u_2_6 <= -1200
Non_overlapping_2_7_constraint_1: s_s_2 - s_s_7 + 27000 u_2_7 <= 26400
Non_overlapping_2_7_constraint_2: - s_s_2 + s_s_7 - 27000 u_2_7 <= -1200
Non_overlapping_2_8_constraint_1: s_s_2 - s_s_8 + 27000 u_2_8 <= 26400
Non_overlapping_2_8_constraint_2: - s_s_2 + s_s_8 - 27000 u_2_8 <= -1200
Non_overlapping_2_9_constraint_1: s_s_2 - s_s_9 + 27000 u_2_9 <= 26400
Non_overlapping_2_9_constraint_2: - s_s_2 + s_s_9 - 27000 u_2_9 <= -1200
Non_overlapping_3_0_constraint_1: - s_s_0 + s_s_3 + 27000 u_3_0 <= 26400
Non_overlapping_3_0_constraint_2: s_s_0 - s_s_3 - 27000 u_3_0 <= -600
Non_overlapping_3_10_constraint_1: - s_s_10 + s_s_3 + 27000 u_3_10 <= 26400
Non_overlapping_3_10_constraint_2: s_s_10 - s_s_3 - 27000 u_3_10 <= -1200
Non_overlapping_3_11_constraint_1: - s_s_11 + s_s_3 + 27000 u_3_11 <= 26400
Non_overlapping_3_11_constraint_2: s_s_11 - s_s_3 - 27000 u_3_11 <= -1800
Non_overlapping_3_12_constraint_1: - s_s_12 + s_s_3 + 27000 u_3_12 <= 26400
Non_overlapping_3_12_constraint_2: s_s_12 - s_s_3 - 27000 u_3_12 <= -1800
Non_overlapping_3_13_constraint_1: - s_s_13 + s_s_3 + 27000 u_3_13 <= 26400
Non_overlapping_3_13_constraint_2: s_s_13 - s_s_3 - 27000 u_3_13 <= -1800
Non_overlapping_3_14_constraint_1: - s_s_14 + s_s_3 + 27000 u_3_14 <= 26400
Non_overlapping_3_14_constraint_2: s_s_14 - s_s_3 - 27000 u_3_14 <= -1800
Non_overlapping_3_15_constraint_1: - s_s_15 + s_s_3 + 27000 u_3_15 <= 26400
Non_overlapping_3_15_constraint_2: s_s_15 - s_s_3 - 27000 u_3_15 <= -2400
Non_overlapping_3_16_constraint_1: - s_s_16 + s_s_3 + 27000 u_3_16 <= 26400
Non_overlapping_3_16_constraint_2: s_s_16 - s_s_3 - 27000 u_3_16 <= -2400
Non_overlapping_3_1_constraint_1: - s_s_1 + s_s_3 + 27000 u_3_1 <= 26400
Non_overlapping_3_1_constraint_2: s_s_1 - s_s_3 - 27000 u_3_1 <= -600
Non_overlapping_3_2_constraint_1: - s_s_2 + s_s_3 + 27000 u_3_2 <= 26400
Non_overlapping_3_2_constraint_2: s_s_2 - s_s_3 - 27000 u_3_2 <= -600
Non_overlapping_3_4_constraint_1: s_s_3 - s_s_4 + 27000 u_3_4 <= 26400
Non_overlapping_3_4_constraint_2: - s_s_3 + s_s_4 - 27000 u_3_4 <= -600
Non_overlapping_3_5_constraint_1: s_s_3 - s_s_5 + 27000 u_3_5 <= 26400
Non_overlapping_3_5_constraint_2: - s_s_3 + s_s_5 - 27000 u_3_5 <= -1200
Non_overlapping_3_6_constraint_1: s_s_3 - s_s_6 + 27000 u_3_6 <= 26400
Non_overlapping_3_6_constraint_2: - s_s_3 + s_s_6 - 27000 u_3_6 <= -1200
Non_overlapping_3_7_constraint_1: s_s_3 - s_s_7 + 27000 u_3_7 <= 26400
Non_overlapping_3_7_constraint_2: - s_s_3 + s_s_7 - 27000 u_3_7 <= -1200
Non_overlapping_3_8_constraint_1: s_s_3 - s_s_8 + 27000 u_3_8 <= 26400
Non_overlapping_3_8_constraint_2: - s_s_3 + s_s_8 - 27000 u_3_8 <= -1200
Non_overlapping_3_9_constraint_1: s_s_3 - s_s_9 + 27000 u_3_9 <= 26400
Non_overlapping_3_9_constraint_2: - s_s_3 + s_s_9 - 27000 u_3_9 <= -1200
Non_overlapping_4_0_constraint_1: - s_s_0 + s_s_4 + 27000 u_4_0 <= 26400
Non_overlapping_4_0_constraint_2: s_s_0 - s_s_4 - 27000 u_4_0 <= -600
Non_overlapping_4_10_constraint_1: - s_s_10 + s_s_4 + 27000 u_4_10 <= 26400
Non_overlapping_4_10_constraint_2: s_s_10 - s_s_4 - 27000 u_4_10 <= -1200
Non_overlapping_4_11_constraint_1: - s_s_11 + s_s_4 + 27000 u_4_11 <= 26400
Non_overlapping_4_11_constraint_2: s_s_11 - s_s_4 - 27000 u_4_11 <= -1800
Non_overlapping_4_12_constraint_1: - s_s_12 + s_s_4 + 27000 u_4_12 <= 26400
Non_overlapping_4_12_constraint_2: s_s_12 - s_s_4 - 27000 u_4_12 <= -1800
Non_overlapping_4_13_constraint_1: - s_s_13 + s_s_4 + 27000 u_4_13 <= 26400
Non_overlapping_4_13_constraint_2: s_s_13 - s_s_4 - 27000 u_4_13 <= -1800
Non_overlapping_4_14_constraint_1: - s_s_14 + s_s_4 + 27000 u_4_14 <= 26400
Non_overlapping_4_14_constraint_2: s_s_14 - s_s_4 - 27000 u_4_14 <= -1800
Non_overlapping_4_15_constraint_1: - s_s_15 + s_s_4 + 27000 u_4_15 <= 26400
Non_overlapping_4_15_constraint_2: s_s_15 - s_s_4 - 27000 u_4_15 <= -2400
Non_overlapping_4_16_constraint_1: - s_s_16 + s_s_4 + 27000 u_4_16 <= 26400
Non_overlapping_4_16_constraint_2: s_s_16 - s_s_4 - 27000 u_4_16 <= -2400
Non_overlapping_4_1_constraint_1: - s_s_1 + s_s_4 + 27000 u_4_1 <= 26400
Non_overlapping_4_1_constraint_2: s_s_1 - s_s_4 - 27000 u_4_1 <= -600
Non_overlapping_4_2_constraint_1: - s_s_2 + s_s_4 + 27000 u_4_2 <= 26400
Non_overlapping_4_2_constraint_2: s_s_2 - s_s_4 - 27000 u_4_2 <= -600
Non_overlapping_4_3_constraint_1: - s_s_3 + s_s_4 + 27000 u_4_3 <= 26400
Non_overlapping_4_3_constraint_2: s_s_3 - s_s_4 - 27000 u_4_3 <= -600
Non_overlapping_4_5_constraint_1: s_s_4 - s_s_5 + 27000 u_4_5 <= 26400
Non_overlapping_4_5_constraint_2: - s_s_4 + s_s_5 - 27000 u_4_5 <= -1200
Non_overlapping_4_6_constraint_1: s_s_4 - s_s_6 + 27000 u_4_6 <= 26400
Non_overlapping_4_6_constraint_2: - s_s_4 + s_s_6 - 27000 u_4_6 <= -1200
Non_overlapping_4_7_constraint_1: s_s_4 - s_s_7 + 27000 u_4_7 <= 26400
Non_overlapping_4_7_constraint_2: - s_s_4 + s_s_7 - 27000 u_4_7 <= -1200
Non_overlapping_4_8_constraint_1: s_s_4 - s_s_8 + 27000 u_4_8 <= 26400
Non_overlapping_4_8_constraint_2: - s_s_4 + s_s_8 - 27000 u_4_8 <= -1200
Non_overlapping_4_9_constraint_1: s_s_4 - s_s_9 + 27000 u_4_9 <= 26400
Non_overlapping_4_9_constraint_2: - s_s_4 + s_s_9 - 27000 u_4_9 <= -1200
Non_overlapping_5_0_constraint_1: - s_s_0 + s_s_5 + 27000 u_5_0 <= 25800
Non_overlapping_5_0_constraint_2: s_s_0 - s_s_5 - 27000 u_5_0 <= -600
Non_overlapping_5_10_constraint_1: - s_s_10 + s_s_5 + 27000 u_5_10 <= 25800
Non_overlapping_5_10_constraint_2: s_s_10 - s_s_5 - 27000 u_5_10 <= -1200
Non_overlapping_5_11_constraint_1: - s_s_11 + s_s_5 + 27000 u_5_11 <= 25800
Non_overlapping_5_11_constraint_2: s_s_11 - s_s_5 - 27000 u_5_11 <= -1800
Non_overlapping_5_12_constraint_1: - s_s_12 + s_s_5 + 27000 u_5_12 <= 25800
Non_overlapping_5_12_constraint_2: s_s_12 - s_s_5 - 27000 u_5_12 <= -1800
Non_overlapping_5_13_constraint_1: - s_s_13 + s_s_5 + 27000 u_5_13 <= 25800
Non_overlapping_5_13_constraint_2: s_s_13 - s_s_5 - 27000 u_5_13 <= -1800
Non_overlapping_5_14_constraint_1: - s_s_14 + s_s_5 + 27000 u_5_14 <= 25800
Non_overlapping_5_14_constraint_2: s_s_14 - s_s_5 - 27000 u_5_14 <= -1800
Non_overlapping_5_15_constraint_1: - s_s_15 + s_s_5 + 27000 u_5_15 <= 25800
Non_overlapping_5_15_constraint_2: s_s_15 - s_s_5 - 27000 u_5_15 <= -2400
Non_overlapping_5_16_constraint_1: - s_s_16 + s_s_5 + 27000 u_5_16 <= 25800
Non_overlapping_5_16_constraint_2: s_s_16 - s_s_5 - 27000 u_5_16 <= -2400
Non_overlapping_5_1_constraint_1: - s_s_1 + s_s_5 + 27000 u_5_1 <= 25800
Non_overlapping_5_1_constraint_2: s_s_1 - s_s_5 - 27000 u_5_1 <= -600
Non_overlapping_5_2_constraint_1: - s_s_2 + s_s_5 + 27000 u_5_2 <= 25800
Non_overlapping_5_2_constraint_2: s_s_2 - s_s_5 - 27000 u_5_2 <= -600
Non_overlapping_5_3_constraint_1: - s_s_3 + s_s_5 + 27000 u_5_3 <= 25800
Non_overlapping_5_3_constraint_2: s_s_3 - s_s_5 - 27000 u_5_3 <= -600
Non_overlapping_5_4_constraint_1: - s_s_4 + s_s_5 + 27000 u_5_4 <= 25800
Non_overlapping_5_4_constraint_2: s_s_4 - s_s_5 - 27000 u_5_4 <= -600
Non_overlapping_5_6_constraint_1: s_s_5 - s_s_6 + 27000 u_5_6 <= 25800
Non_overlapping_5_6_constraint_2: - s_s_5 + s_s_6 - 27000 u_5_6 <= -1200
Non_overlapping_5_7_constraint_1: s_s_5 - s_s_7 + 27000 u_5_7 <= 25800
Non_overlapping_5_7_constraint_2: - s_s_5 + s_s_7 - 27000 u_5_7 <= -1200
Non_overlapping_5_8_constraint_1: s_s_5 - s_s_8 + 27000 u_5_8 <= 25800
Non_overlapping_5_8_constraint_2: - s_s_5 + s_s_8 - 27000 u_5_8 <= -1200
Non_overlapping_5_9_constraint_1: s_s_5 - s_s_9 + 27000 u_5_9 <= 25800
Non_overlapping_5_9_constraint_2: - s_s_5 + s_s_9 - 27000 u_5_9 <= -1200
Non_overlapping_6_0_constraint_1: - s_s_0 + s_s_6 + 27000 u_6_0 <= 25800
Non_overlapping_6_0_constraint_2: s_s_0 - s_s_6 - 27000 u_6_0 <= -600
Non_overlapping_6_10_constraint_1: - s_s_10 + s_s_6 + 27000 u_6_10 <= 25800
Non_overlapping_6_10_constraint_2: s_s_10 - s_s_6 - 27000 u_6_10 <= -1200
Non_overlapping_6_11_constraint_1: - s_s_11 + s_s_6 + 27000 u_6_11 <= 25800
Non_overlapping_6_11_constraint_2: s_s_11 - s_s_6 - 27000 u_6_11 <= -1800
Non_overlapping_6_12_constraint_1: - s_s_12 + s_s_6 + 27000 u_6_12 <= 25800
Non_overlapping_6_12_constraint_2: s_s_12 - s_s_6 - 27000 u_6_12 <= -1800
Non_overlapping_6_13_constraint_1: - s_s_13 + s_s_6 + 27000 u_6_13 <= 25800
Non_overlapping_6_13_constraint_2: s_s_13 - s_s_6 - 27000 u_6_13 <= -1800
Non_overlapping_6_14_constraint_1: - s_s_14 + s_s_6 + 27000 u_6_14 <= 25800
Non_overlapping_6_14_constraint_2: s_s_14 - s_s_6 - 27000 u_6_14 <= -1800
Non_overlapping_6_15_constraint_1: - s_s_15 + s_s_6 + 27000 u_6_15 <= 25800
Non_overlapping_6_15_constraint_2: s_s_15 - s_s_6 - 27000 u_6_15 <= -2400
Non_overlapping_6_16_constraint_1: - s_s_16 + s_s_6 + 27000 u_6_16 <= 25800
Non_overlapping_6_16_constraint_2: s_s_16 - s_s_6 - 27000 u_6_16 <= -2400
Non_overlapping_6_1_constraint_1: - s_s_1 + s_s_6 + 27000 u_6_1 <= 25800
Non_overlapping_6_1_constraint_2: s_s_1 - s_s_6 - 27000 u_6_1 <= -600
Non_overlapping_6_2_constraint_1: - s_s_2 + s_s_6 + 27000 u_6_2 <= 25800
Non_overlapping_6_2_constraint_2: s_s_2 - s_s_6 - 27000 u_6_2 <= -600
Non_overlapping_6_3_constraint_1: - s_s_3 + s_s_6 + 27000 u_6_3 <= 25800
Non_overlapping_6_3_constraint_2: s_s_3 - s_s_6 - 27000 u_6_3 <= -600
Non_overlapping_6_4_constraint_1: - s_s_4 + s_s_6 + 27000 u_6_4 <= 25800
Non_overlapping_6_4_constraint_2: s_s_4 - s_s_6 - 27000 u_6_4 <= -600
Non_overlapping_6_5_constraint_1: - s_s_5 + s_s_6 + 27000 u_6_5 <= 25800
Non_overlapping_6_5_constraint_2: s_s_5 - s_s_6 - 27000 u_6_5 <= -1200
Non_overlapping_6_7_constraint_1: s_s_6 - s_s_7 + 27000 u_6_7 <= 25800
Non_overlapping_6_7_constraint_2: - s_s_6 + s_s_7 - 27000 u_6_7 <= -1200
Non_overlapping_6_8_constraint_1: s_s_6 - s_s_8 + 27000 u_6_8 <= 25800
Non_overlapping_6_8_constraint_2: - s_s_6 + s_s_8 - 27000 u_6_8 <= -1200
Non_overlapping_6_9_constraint_1: s_s_6 - s_s_9 + 27000 u_6_9 <= 25800
Non_overlapping_6_9_constraint_2: - s_s_6 + s_s_9 - 27000 u_6_9 <= -1200
Non_overlapping_7_0_constraint_1: - s_s_0 + s_s_7 + 27000 u_7_0 <= 25800
Non_overlapping_7_0_constraint_2: s_s_0 - s_s_7 - 27000 u_7_0 <= -600
Non_overlapping_7_10_constraint_1: - s_s_10 + s_s_7 + 27000 u_7_10 <= 25800
Non_overlapping_7_10_constraint_2: s_s_10 - s_s_7 - 27000 u_7_10 <= -1200
Non_overlapping_7_11_constraint_1: - s_s_11 + s_s_7 + 27000 u_7_11 <= 25800
Non_overlapping_7_11_constraint_2: s_s_11 - s_s_7 - 27000 u_7_11 <= -1800
Non_overlapping_7_12_constraint_1: - s_s_12 + s_s_7 + 27000 u_7_12 <= 25800
Non_overlapping_7_12_constraint_2: s_s_12 - s_s_7 - 27000 u_7_12 <= -1800
Non_overlapping_7_13_constraint_1: - s_s_13 + s_s_7 + 27000 u_7_13 <= 25800
Non_overlapping_7_13_constraint_2: s_s_13 - s_s_7 - 27000 u_7_13 <= -1800
Non_overlapping_7_14_constraint_1: - s_s_14 + s_s_7 + 27000 u_7_14 <= 25800
Non_overlapping_7_14_constraint_2: s_s_14 - s_s_7 - 27000 u_7_14 <= -1800
Non_overlapping_7_15_constraint_1: - s_s_15 + s_s_7 + 27000 u_7_15 <= 25800
Non_overlapping_7_15_constraint_2: s_s_15 - s_s_7 - 27000 u_7_15 <= -2400
Non_overlapping_7_16_constraint_1: - s_s_16 + s_s_7 + 27000 u_7_16 <= 25800
Non_overlapping_7_16_constraint_2: s_s_16 - s_s_7 - 27000 u_7_16 <= -2400
Non_overlapping_7_1_constraint_1: - s_s_1 + s_s_7 + 27000 u_7_1 <= 25800
Non_overlapping_7_1_constraint_2: s_s_1 - s_s_7 - 27000 u_7_1 <= -600
Non_overlapping_7_2_constraint_1: - s_s_2 + s_s_7 + 27000 u_7_2 <= 25800
Non_overlapping_7_2_constraint_2: s_s_2 - s_s_7 - 27000 u_7_2 <= -600
Non_overlapping_7_3_constraint_1: - s_s_3 + s_s_7 + 27000 u_7_3 <= 25800
Non_overlapping_7_3_constraint_2: s_s_3 - s_s_7 - 27000 u_7_3 <= -600
Non_overlapping_7_4_constraint_1: - s_s_4 + s_s_7 + 27000 u_7_4 <= 25800
Non_overlapping_7_4_constraint_2: s_s_4 - s_s_7 - 27000 u_7_4 <= -600
Non_overlapping_7_5_constraint_1: - s_s_5 + s_s_7 + 27000 u_7_5 <= 25800
Non_overlapping_7_5_constraint_2: s_s_5 - s_s_7 - 27000 u_7_5 <= -1200
Non_overlapping_7_6_constraint_1: - s_s_6 + s_s_7 + 27000 u_7_6 <= 25800
Non_overlapping_7_6_constraint_2: s_s_6 - s_s_7 - 27000 u_7_6 <= -1200
Non_overlapping_7_8_constraint_1: s_s_7 - s_s_8 + 27000 u_7_8 <= 25800
Non_overlapping_7_8_constraint_2: - s_s_7 + s_s_8 - 27000 u_7_8 <= -1200
Non_overlapping_7_9_constraint_1: s_s_7 - s_s_9 + 27000 u_7_9 <= 25800
Non_overlapping_7_9_constraint_2: - s_s_7 + s_s_9 - 27000 u_7_9 <= -1200
Non_overlapping_8_0_constraint_1: - s_s_0 + s_s_8 + 27000 u_8_0 <= 25800
Non_overlapping_8_0_constraint_2: s_s_0 - s_s_8 - 27000 u_8_0 <= -600
Non_overlapping_8_10_constraint_1: - s_s_10 + s_s_8 + 27000 u_8_10 <= 25800
Non_overlapping_8_10_constraint_2: s_s_10 - s_s_8 - 27000 u_8_10 <= -1200
Non_overlapping_8_11_constraint_1: - s_s_11 + s_s_8 + 27000 u_8_11 <= 25800
Non_overlapping_8_11_constraint_2: s_s_11 - s_s_8 - 27000 u_8_11 <= -1800
Non_overlapping_8_12_constraint_1: - s_s_12 + s_s_8 + 27000 u_8_12 <= 25800
Non_overlapping_8_12_constraint_2: s_s_12 - s_s_8 - 27000 u_8_12 <= -1800
Non_overlapping_8_13_constraint_1: - s_s_13 + s_s_8 + 27000 u_8_13 <= 25800
Non_overlapping_8_13_constraint_2: s_s_13 - s_s_8 - 27000 u_8_13 <= -1800
Non_overlapping_8_14_constraint_1: - s_s_14 + s_s_8 + 27000 u_8_14 <= 25800
Non_overlapping_8_14_constraint_2: s_s_14 - s_s_8 - 27000 u_8_14 <= -1800
Non_overlapping_8_15_constraint_1: - s_s_15 + s_s_8 + 27000 u_8_15 <= 25800
Non_overlapping_8_15_constraint_2: s_s_15 - s_s_8 - 27000 u_8_15 <= -2400
Non_overlapping_8_16_constraint_1: - s_s_16 + s_s_8 + 27000 u_8_16 <= 25800
Non_overlapping_8_16_constraint_2: s_s_16 - s_s_8 - 27000 u_8_16 <= -2400
Non_overlapping_8_1_constraint_1: - s_s_1 + s_s_8 + 27000 u_8_1 <= 25800
Non_overlapping_8_1_constraint_2: s_s_1 - s_s_8 - 27000 u_8_1 <= -600
Non_overlapping_8_2_constraint_1: - s_s_2 + s_s_8 + 27000 u_8_2 <= 25800
Non_overlapping_8_2_constraint_2: s_s_2 - s_s_8 - 27000 u_8_2 <= -600
Non_overlapping_8_3_constraint_1: - s_s_3 + s_s_8 + 27000 u_8_3 <= 25800
Non_overlapping_8_3_constraint_2: s_s_3 - s_s_8 - 27000 u_8_3 <= -600
Non_overlapping_8_4_constraint_1: - s_s_4 + s_s_8 + 27000 u_8_4 <= 25800
Non_overlapping_8_4_constraint_2: s_s_4 - s_s_8 - 27000 u_8_4 <= -600
Non_overlapping_8_5_constraint_1: - s_s_5 + s_s_8 + 27000 u_8_5 <= 25800
Non_overlapping_8_5_constraint_2: s_s_5 - s_s_8 - 27000 u_8_5 <= -1200
Non_overlapping_8_6_constraint_1: - s_s_6 + s_s_8 + 27000 u_8_6 <= 25800
Non_overlapping_8_6_constraint_2: s_s_6 - s_s_8 - 27000 u_8_6 <= -1200
Non_overlapping_8_7_constraint_1: - s_s_7 + s_s_8 + 27000 u_8_7 <= 25800
Non_overlapping_8_7_constraint_2: s_s_7 - s_s_8 - 27000 u_8_7 <= -1200
Non_overlapping_8_9_constraint_1: s_s_8 - s_s_9 + 27000 u_8_9 <= 25800
Non_overlapping_8_9_constraint_2: - s_s_8 + s_s_9 - 27000 u_8_9 <= -1200
Non_overlapping_9_0_constraint_1: - s_s_0 + s_s_9 + 27000 u_9_0 <= 25800
Non_overlapping_9_0_constraint_2: s_s_0 - s_s_9 - 27000 u_9_0 <= -600
Non_overlapping_9_10_constraint_1: - s_s_10 + s_s_9 + 27000 u_9_10 <= 25800
Non_overlapping_9_10_constraint_2: s_s_10 - s_s_9 - 27000 u_9_10 <= -1200
Non_overlapping_9_11_constraint_1: - s_s_11 + s_s_9 + 27000 u_9_11 <= 25800
Non_overlapping_9_11_constraint_2: s_s_11 - s_s_9 - 27000 u_9_11 <= -1800
Non_overlapping_9_12_constraint_1: - s_s_12 + s_s_9 + 27000 u_9_12 <= 25800
Non_overlapping_9_12_constraint_2: s_s_12 - s_s_9 - 27000 u_9_12 <= -1800
Non_overlapping_9_13_constraint_1: - s_s_13 + s_s_9 + 27000 u_9_13 <= 25800
Non_overlapping_9_13_constraint_2: s_s_13 - s_s_9 - 27000 u_9_13 <= -1800
Non_overlapping_9_14_constraint_1: - s_s_14 + s_s_9 + 27000 u_9_14 <= 25800
Non_overlapping_9_14_constraint_2: s_s_14 - s_s_9 - 27000 u_9_14 <= -1800
Non_overlapping_9_15_constraint_1: - s_s_15 + s_s_9 + 27000 u_9_15 <= 25800
Non_overlapping_9_15_constraint_2: s_s_15 - s_s_9 - 27000 u_9_15 <= -2400
Non_overlapping_9_16_constraint_1: - s_s_16 + s_s_9 + 27000 u_9_16 <= 25800
Non_overlapping_9_16_constraint_2: s_s_16 - s_s_9 - 27000 u_9_16 <= -2400
Non_overlapping_9_1_constraint_1: - s_s_1 + s_s_9 + 27000 u_9_1 <= 25800
Non_overlapping_9_1_constraint_2: s_s_1 - s_s_9 - 27000 u_9_1 <= -600
Non_overlapping_9_2_constraint_1: - s_s_2 + s_s_9 + 27000 u_9_2 <= 25800
Non_overlapping_9_2_constraint_2: s_s_2 - s_s_9 - 27000 u_9_2 <= -600
Non_overlapping_9_3_constraint_1: - s_s_3 + s_s_9 + 27000 u_9_3 <= 25800
Non_overlapping_9_3_constraint_2: s_s_3 - s_s_9 - 27000 u_9_3 <= -600
Non_overlapping_9_4_constraint_1: - s_s_4 + s_s_9 + 27000 u_9_4 <= 25800
Non_overlapping_9_4_constraint_2: s_s_4 - s_s_9 - 27000 u_9_4 <= -600
Non_overlapping_9_5_constraint_1: - s_s_5 + s_s_9 + 27000 u_9_5 <= 25800
Non_overlapping_9_5_constraint_2: s_s_5 - s_s_9 - 27000 u_9_5 <= -1200
Non_overlapping_9_6_constraint_1: - s_s_6 + s_s_9 + 27000 u_9_6 <= 25800
Non_overlapping_9_6_constraint_2: s_s_6 - s_s_9 - 27000 u_9_6 <= -1200
Non_overlapping_9_7_constraint_1: - s_s_7 + s_s_9 + 27000 u_9_7 <= 25800
Non_overlapping_9_7_constraint_2: s_s_7 - s_s_9 - 27000 u_9_7 <= -1200
Non_overlapping_9_8_constraint_1: - s_s_8 + s_s_9 + 27000 u_9_8 <= 25800
Non_overlapping_9_8_constraint_2: s_s_8 - s_s_9 - 27000 u_9_8 <= -1200
Take_out_before_delivery_0: s_s_0 <= 9000
Take_out_before_delivery_1: s_s_1 <= 26400
Take_out_before_delivery_10: s_s_10 <= 24000
Take_out_before_delivery_11: s_s_11 <= 22200
Take_out_before_delivery_12: s_s_12 <= 16800
Take_out_before_delivery_13: s_s_13 <= 22200
Take_out_before_delivery_14: s_s_14 <= 13200
Take_out_before_delivery_15: s_s_15 <= 6600
Take_out_before_delivery_16: s_s_16 <= 3000
Take_out_before_delivery_2: s_s_2 <= 0
Take_out_before_delivery_3: s_s_3 <= 4800
Take_out_before_delivery_4: s_s_4 <= 3600
Take_out_before_delivery_5: s_s_5 <= 9600
Take_out_before_delivery_6: s_s_6 <= 13800
Take_out_before_delivery_7: s_s_7 <= 13200
Take_out_before_delivery_8: s_s_8 <= 19200
Take_out_before_delivery_9: s_s_9 <= 23400
Upper_bound_on_start_0: s_s_0 - start >= 0
Upper_bound_on_start_1: s_s_1 - start >= 0
Upper_bound_on_start_10: s_s_10 - start >= 0
Upper_bound_on_start_11: s_s_11 - start >= 0
Upper_bound_on_start_12: s_s_12 - start >= 0
Upper_bound_on_start_13: s_s_13 - start >= 0
Upper_bound_on_start_14: s_s_14 - start >= 0
Upper_bound_on_start_15: s_s_15 - start >= 0
Upper_bound_on_start_16: s_s_16 - start >= 0
Upper_bound_on_start_2: s_s_2 - start >= 0
Upper_bound_on_start_3: s_s_3 - start >= 0
Upper_bound_on_start_4: s_s_4 - start >= 0
Upper_bound_on_start_5: s_s_5 - start >= 0
Upper_bound_on_start_6: s_s_6 - start >= 0
Upper_bound_on_start_7: s_s_7 - start >= 0
Upper_bound_on_start_8: s_s_8 - start >= 0
Upper_bound_on_start_9: s_s_9 - start >= 0
Bounds
 0 <= end
 0 <= s_s_0 <= 86400
 0 <= s_s_1 <= 86400
 0 <= s_s_10 <= 86400
 0 <= s_s_11 <= 86400
 0 <= s_s_12 <= 86400
 0 <= s_s_13 <= 86400
 0 <= s_s_14 <= 86400
 0 <= s_s_15 <= 86400
 0 <= s_s_16 <= 86400
 0 <= s_s_2 <= 86400
 0 <= s_s_3 <= 86400
 0 <= s_s_4 <= 86400
 0 <= s_s_5 <= 86400
 0 <= s_s_6 <= 86400
 0 <= s_s_7 <= 86400
 0 <= s_s_8 <= 86400
 0 <= s_s_9 <= 86400
 0 <= start
Generals
end
s_s_0
s_s_1
s_s_10
s_s_11
s_s_12
s_s_13
s_s_14
s_s_15
s_s_16
s_s_2
s_s_3
s_s_4
s_s_5
s_s_6
s_s_7
s_s_8
s_s_9
start
Binaries
u_0_1
u_0_10
u_0_11
u_0_12
u_0_13
u_0_14
u_0_15
u_0_16
u_0_2
u_0_3
u_0_4
u_0_5
u_0_6
u_0_7
u_0_8
u_0_9
u_10_0
u_10_1
u_10_11
u_10_12
u_10_13
u_10_14
u_10_15
u_10_16
u_10_2
u_10_3
u_10_4
u_10_5
u_10_6
u_10_7
u_10_8
u_10_9
u_11_0
u_11_1
u_11_10
u_11_12
u_11_13
u_11_14
u_11_15
u_11_16
u_11_2
u_11_3
u_11_4
u_11_5
u_11_6
u_11_7
u_11_8
u_11_9
u_12_0
u_12_1
u_12_10
u_12_11
u_12_13
u_12_14
u_12_15
u_12_16
u_12_2
u_12_3
u_12_4
u_12_5
u_12_6
u_12_7
u_12_8
u_12_9
u_13_0
u_13_1
u_13_10
u_13_11
u_13_12
u_13_14
u_13_15
u_13_16
u_13_2
u_13_3
u_13_4
u_13_5
u_13_6
u_13_7
u_13_8
u_13_9
u_14_0
u_14_1
u_14_10
u_14_11
u_14_12
u_14_13
u_14_15
u_14_16
u_14_2
u_14_3
u_14_4
u_14_5
u_14_6
u_14_7
u_14_8
u_14_9
u_15_0
u_15_1
u_15_10
u_15_11
u_15_12
u_15_13
u_15_14
u_15_16
u_15_2
u_15_3
u_15_4
u_15_5
u_15_6
u_15_7
u_15_8
u_15_9
u_16_0
u_16_1
u_16_10
u_16_11
u_16_12
u_16_13
u_16_14
u_16_15
u_16_2
u_16_3
u_16_4
u_16_5
u_16_6
u_16_7
u_16_8
u_16_9
u_1_0
u_1_10
u_1_11
u_1_12
u_1_13
u_1_14
u_1_15
u_1_16
u_1_2
u_1_3
u_1_4
u_1_5
u_1_6
u_1_7
u_1_8
u_1_9
u_2_0
u_2_1
u_2_10
u_2_11
u_2_12
u_2_13
u_2_14
u_2_15
u_2_16
u_2_3
u_2_4
u_2_5
u_2_6
u_2_7
u_2_8
u_2_9
u_3_0
u_3_1
u_3_10
u_3_11
u_3_12
u_3_13
u_3_14
u_3_15
u_3_16
u_3_2
u_3_4
u_3_5
u_3_6
u_3_7
u_3_8
u_3_9
u_4_0
u_4_1
u_4_10
u_4_11
u_4_12
u_4_13
u_4_14
u_4_15
u_4_16
u_4_2
u_4_3
u_4_5
u_4_6
u_4_7
u_4_8
u_4_9
u_5_0
u_5_1
u_5_10
u_5_11
u_5_12
u_5_13
u_5_14
u_5_15
u_5_16
u_5_2
u_5_3
u_5_4
u_5_6
u_5_7
u_5_8
u_5_9
u_6_0
u_6_1
u_6_10
u_6_11
u_6_12
u_6_13
u_6_14
u_6_15
u_6_16
u_6_2
u_6_3
u_6_4
u_6_5
u_6_7
u_6_8
u_6_9
u_7_0
u_7_1
u_7_10
u_7_11
u_7_12
u_7_13
u_7_14
u_7_15
u_7_16
u_7_2
u_7_3
u_7_4
u_7_5
u_7_6
u_7_8
u_7_9
u_8_0
u_8_1
u_8_10
u_8_11
u_8_12
u_8_13
u_8_14
u_8_15
u_8_16
u_8_2
u_8_3
u_8_4
u_8_5
u_8_6
u_8_7
u_8_9
u_9_0
u_9_1
u_9_10
u_9_11
u_9_12
u_9_13
u_9_14
u_9_15
u_9_16
u_9_2
u_9_3
u_9_4
u_9_5
u_9_6
u_9_7
u_9_8
End

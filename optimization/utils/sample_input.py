import numpy as np

points = np.array([
    (110.14, 0.00),
    (105.83, 1.33),
    (101.50, 2.55),
    (97.16, 3.66),
    (92.81, 4.67),
    (88.45, 5.56),
    (84.08, 6.35),
    (79.71, 7.03),
    (75.34, 7.60),
    (70.96, 8.06),
    (66.58, 8.41),
    (62.21, 8.65),
    (57.84, 8.79),
    (57.55, 9.49),
    (61.30, 10.90),
    (65.02, 12.40),
    (68.72, 14.01),
    (72.38, 15.70),
    (76.01, 17.49),
    (79.61, 19.38),
    (83.17, 21.35),
    (86.69, 23.42),
    (90.18, 25.59),
    (93.62, 27.84),
    (97.02, 30.18),
    (100.38, 32.61),
    (100.17, 33.94),
    (99.94, 35.27),
    (99.70, 36.60),
    (99.44, 37.93),
    (99.17, 39.26),
    (98.87, 40.59),
    (98.56, 41.92),
    (98.23, 43.24),
    (97.88, 44.57),
    (97.51, 45.89),
    (97.13, 47.21),
    (96.73, 48.52),
    (94.43, 48.87),
    (90.28, 48.17),
    (86.16, 47.37),
    (82.08, 46.48),
    (78.04, 45.49),
    (74.03, 44.41),
    (70.06, 43.24),
    (66.13, 41.97),
    (62.25, 40.61),
    (58.41, 39.16),
    (54.62, 37.62),
    (50.88, 36.00),
    (47.18, 34.28),
    (46.75, 34.87),
    (46.31, 35.46),
    (45.86, 36.04),
    (45.40, 36.61),
    (44.94, 37.18),
    (44.47, 37.74),
    (43.99, 38.29),
    (43.51, 38.84),
    (43.01, 39.39),
    (42.52, 39.93),
    (42.01, 40.46),
    (41.50, 40.98),
    (42.39, 42.92),
    (44.62, 46.33),
    (46.77, 49.81),
    (48.84, 53.34),
    (50.83, 56.93),
    (52.74, 60.59),
    (54.56, 64.29),
    (56.30, 68.06),
    (57.95, 71.87),
    (59.52, 75.74),
    (60.99, 79.66),
    (62.38, 83.63),
    (63.67, 87.64),
    (62.44, 88.25),
    (61.20, 88.84),
    (59.95, 89.42),
    (58.70, 89.98),
    (57.45, 90.52),
    (56.19, 91.05),
    (54.93, 91.56),
    (53.66, 92.05),
    (52.39, 92.52),
    (51.11, 92.97),
    (49.84, 93.41),
    (48.56, 93.83),
    (46.44, 92.58),
    (43.55, 89.62),
    (40.75, 86.60),
    (38.03, 83.52),
    (35.39, 80.39),
    (32.84, 77.21),
    (30.37, 73.97),
    (27.99, 70.69),
    (25.70, 67.36),
    (23.49, 63.99),
    (21.38, 60.57),
    (19.35, 57.10),
    (17.42, 53.60),
    (18.02, 57.92),
    (18.52, 62.27),
    (18.91, 66.63),
    (19.19, 71.02),
    (19.37, 75.42),
    (19.43, 79.84),
    (19.39, 84.27),
    (19.25, 88.71),
    (18.99, 93.16),
    (18.62, 97.62),
    (18.15, 102.08),
    (17.56, 106.55),
    (16.55, 108.89),
    (15.18, 109.09),
    (13.80, 109.27),
    (12.43, 109.43),
    (11.05, 109.58),
    (9.68, 109.71),
    (8.30, 109.83),
    (6.92, 109.92),
    (5.53, 110.00),
    (4.15, 110.06),
    (2.77, 110.10),
    (1.38, 110.13),
    (0.00, 110.14),
    (-1.33, 105.83),
    (-2.55, 101.50),
    (-3.66, 97.16),
    (-4.67, 92.81),
    (-5.56, 88.45),
    (-6.35, 84.08),
    (-7.03, 79.71),
    (-7.60, 75.34),
    (-8.06, 70.96),
    (-8.41, 66.58),
    (-8.65, 62.21),
    (-8.79, 57.84),
    (-9.49, 57.55),
    (-10.90, 61.30),
    (-12.40, 65.02),
    (-14.01, 68.72),
    (-15.70, 72.38),
    (-17.49, 76.01),
    (-19.38, 79.61),
    (-21.35, 83.17),
    (-23.42, 86.69),
    (-25.59, 90.18),
    (-27.84, 93.62),
    (-30.18, 97.02),
    (-32.61, 100.38),
    (-33.94, 100.17),
    (-35.27, 99.94),
    (-36.60, 99.70),
    (-37.93, 99.44),
    (-39.26, 99.17),
    (-40.59, 98.87),
    (-41.92, 98.56),
    (-43.24, 98.23),
    (-44.57, 97.88),
    (-45.89, 97.51),
    (-47.21, 97.13),
    (-48.52, 96.73),
    (-48.87, 94.43),
    (-48.17, 90.28),
    (-47.37, 86.16),
    (-46.48, 82.08),
    (-45.49, 78.04),
    (-44.41, 74.03),
    (-43.24, 70.06),
    (-41.97, 66.13),
    (-40.61, 62.25),
    (-39.16, 58.41),
    (-37.62, 54.62),
    (-36.00, 50.88),
    (-34.28, 47.18),
    (-34.87, 46.75),
    (-35.46, 46.31),
    (-36.04, 45.86),
    (-36.61, 45.40),
    (-37.18, 44.94),
    (-37.74, 44.47),
    (-38.29, 43.99),
    (-38.84, 43.51),
    (-39.39, 43.01),
    (-39.93, 42.52),
    (-40.46, 42.01),
    (-40.98, 41.50),
    (-42.92, 42.39),
    (-46.33, 44.62),
    (-49.81, 46.77),
    (-53.34, 48.84),
    (-56.93, 50.83),
    (-60.59, 52.74),
    (-64.29, 54.56),
    (-68.06, 56.30),
    (-71.87, 57.95),
    (-75.74, 59.52),
    (-79.66, 60.99),
    (-83.63, 62.38),
    (-87.64, 63.67),
    (-88.25, 62.44),
    (-88.84, 61.20),
    (-89.42, 59.95),
    (-89.98, 58.70),
    (-90.52, 57.45),
    (-91.05, 56.19),
    (-91.56, 54.93),
    (-92.05, 53.66),
    (-92.52, 52.39),
    (-92.97, 51.11),
    (-93.41, 49.84),
    (-93.83, 48.56),
    (-92.58, 46.44),
    (-89.62, 43.55),
    (-86.60, 40.75),
    (-83.52, 38.03),
    (-80.39, 35.39),
    (-77.21, 32.84),
    (-73.97, 30.37),
    (-70.69, 27.99),
    (-67.36, 25.70),
    (-63.99, 23.49),
    (-60.57, 21.38),
    (-57.10, 19.35),
    (-53.60, 17.42),
    (-57.92, 18.02),
    (-62.27, 18.52),
    (-66.63, 18.91),
    (-71.02, 19.19),
    (-75.42, 19.37),
    (-79.84, 19.43),
    (-84.27, 19.39),
    (-88.71, 19.25),
    (-93.16, 18.99),
    (-97.62, 18.62),
    (-102.08, 18.15),
    (-106.55, 17.56),
    (-108.89, 16.55),
    (-109.09, 15.18),
    (-109.27, 13.80),
    (-109.43, 12.43),
    (-109.58, 11.05),
    (-109.71, 9.68),
    (-109.83, 8.30),
    (-109.92, 6.92),
    (-110.00, 5.53),
    (-110.06, 4.15),
    (-110.10, 2.77),
    (-110.13, 1.38),
    (-110.14, 0.00),
    (-105.83, -1.33),
    (-101.50, -2.55),
    (-97.16, -3.66),
    (-92.81, -4.67),
    (-88.45, -5.56),
    (-84.08, -6.35),
    (-79.71, -7.03),
    (-75.34, -7.60),
    (-70.96, -8.06),
    (-66.58, -8.41),
    (-62.21, -8.65),
    (-57.84, -8.79),
    (-57.55, -9.49),
    (-61.30, -10.90),
    (-65.02, -12.40),
    (-68.72, -14.01),
    (-72.38, -15.70),
    (-76.01, -17.49),
    (-79.61, -19.38),
    (-83.17, -21.35),
    (-86.69, -23.42),
    (-90.18, -25.59),
    (-93.62, -27.84),
    (-97.02, -30.18),
    (-100.38, -32.61),
    (-100.17, -33.94),
    (-99.94, -35.27),
    (-99.70, -36.60),
    (-99.44, -37.93),
    (-99.17, -39.26),
    (-98.87, -40.59),
    (-98.56, -41.92),
    (-98.23, -43.24),
    (-97.88, -44.57),
    (-97.51, -45.89),
    (-97.13, -47.21),
    (-96.73, -48.52),
    (-94.43, -48.87),
    (-90.28, -48.17),
    (-86.16, -47.37),
    (-82.08, -46.48),
    (-78.04, -45.49),
    (-74.03, -44.41),
    (-70.06, -43.24),
    (-66.13, -41.97),
    (-62.25, -40.61),
    (-58.41, -39.16),
    (-54.62, -37.62),
    (-50.88, -36.00),
    (-47.18, -34.28),
    (-46.75, -34.87),
    (-46.31, -35.46),
    (-45.86, -36.04),
    (-45.40, -36.61),
    (-44.94, -37.18),
    (-44.47, -37.74),
    (-43.99, -38.29),
    (-43.51, -38.84),
    (-43.01, -39.39),
    (-42.52, -39.93),
    (-42.01, -40.46),
    (-41.50, -40.98),
    (-42.39, -42.92),
    (-44.62, -46.33),
    (-46.77, -49.81),
    (-48.84, -53.34),
    (-50.83, -56.93),
    (-52.74, -60.59),
    (-54.56, -64.29),
    (-56.30, -68.06),
    (-57.95, -71.87),
    (-59.52, -75.74),
    (-60.99, -79.66),
    (-62.38, -83.63),
    (-63.67, -87.64),
    (-62.44, -88.25),
    (-61.20, -88.84),
    (-59.95, -89.42),
    (-58.70, -89.98),
    (-57.45, -90.52),
    (-56.19, -91.05),
    (-54.93, -91.56),
    (-53.66, -92.05),
    (-52.39, -92.52),
    (-51.11, -92.97),
    (-49.84, -93.41),
    (-48.56, -93.83),
    (-46.44, -92.58),
    (-43.55, -89.62),
    (-40.75, -86.60),
    (-38.03, -83.52),
    (-35.39, -80.39),
    (-32.84, -77.21),
    (-30.37, -73.97),
    (-27.99, -70.69),
    (-25.70, -67.36),
    (-23.49, -63.99),
    (-21.38, -60.57),
    (-19.35, -57.10),
    (-17.42, -53.60),
    (-18.02, -57.92),
    (-18.52, -62.27),
    (-18.91, -66.63),
    (-19.19, -71.02),
    (-19.37, -75.42),
    (-19.43, -79.84),
    (-19.39, -84.27),
    (-19.25, -88.71),
    (-18.99, -93.16),
    (-18.62, -97.62),
    (-18.15, -102.08),
    (-17.56, -106.55),
    (-16.55, -108.89),
    (-15.18, -109.09),
    (-13.80, -109.27),
    (-12.43, -109.43),
    (-11.05, -109.58),
    (-9.68, -109.71),
    (-8.30, -109.83),
    (-6.92, -109.92),
    (-5.53, -110.00),
    (-4.15, -110.06),
    (-2.77, -110.10),
    (-1.38, -110.13),
    (-0.00, -110.14),
    (1.33, -105.83),
    (2.55, -101.50),
    (3.66, -97.16),
    (4.67, -92.81),
    (5.56, -88.45),
    (6.35, -84.08),
    (7.03, -79.71),
    (7.60, -75.34),
    (8.06, -70.96),
    (8.41, -66.58),
    (8.65, -62.21),
    (8.79, -57.84),
    (9.49, -57.55),
    (10.90, -61.30),
    (12.40, -65.02),
    (14.01, -68.72),
    (15.70, -72.38),
    (17.49, -76.01),
    (19.38, -79.61),
    (21.35, -83.17),
    (23.42, -86.69),
    (25.59, -90.18),
    (27.84, -93.62),
    (30.18, -97.02),
    (32.61, -100.38),
    (33.94, -100.17),
    (35.27, -99.94),
    (36.60, -99.70),
    (37.93, -99.44),
    (39.26, -99.17),
    (40.59, -98.87),
    (41.92, -98.56),
    (43.24, -98.23),
    (44.57, -97.88),
    (45.89, -97.51),
    (47.21, -97.13),
    (48.52, -96.73),
    (48.87, -94.43),
    (48.17, -90.28),
    (47.37, -86.16),
    (46.48, -82.08),
    (45.49, -78.04),
    (44.41, -74.03),
    (43.24, -70.06),
    (41.97, -66.13),
    (40.61, -62.25),
    (39.16, -58.41),
    (37.62, -54.62),
    (36.00, -50.88),
    (34.28, -47.18),
    (34.87, -46.75),
    (35.46, -46.31),
    (36.04, -45.86),
    (36.61, -45.40),
    (37.18, -44.94),
    (37.74, -44.47),
    (38.29, -43.99),
    (38.84, -43.51),
    (39.39, -43.01),
    (39.93, -42.52),
    (40.46, -42.01),
    (40.98, -41.50),
    (42.92, -42.39),
    (46.33, -44.62),
    (49.81, -46.77),
    (53.34, -48.84),
    (56.93, -50.83),
    (60.59, -52.74),
    (64.29, -54.56),
    (68.06, -56.30),
    (71.87, -57.95),
    (75.74, -59.52),
    (79.66, -60.99),
    (83.63, -62.38),
    (87.64, -63.67),
    (88.25, -62.44),
    (88.84, -61.20),
    (89.42, -59.95),
    (89.98, -58.70),
    (90.52, -57.45),
    (91.05, -56.19),
    (91.56, -54.93),
    (92.05, -53.66),
    (92.52, -52.39),
    (92.97, -51.11),
    (93.41, -49.84),
    (93.83, -48.56),
    (92.58, -46.44),
    (89.62, -43.55),
    (86.60, -40.75),
    (83.52, -38.03),
    (80.39, -35.39),
    (77.21, -32.84),
    (73.97, -30.37),
    (70.69, -27.99),
    (67.36, -25.70),
    (63.99, -23.49),
    (60.57, -21.38),
    (57.10, -19.35),
    (53.60, -17.42),
    (57.92, -18.02),
    (62.27, -18.52),
    (66.63, -18.91),
    (71.02, -19.19),
    (75.42, -19.37),
    (79.84, -19.43),
    (84.27, -19.39),
    (88.71, -19.25),
    (93.16, -18.99),
    (97.62, -18.62),
    (102.08, -18.15),
    (106.55, -17.56),
    (108.89, -16.55),
    (109.09, -15.18),
    (109.27, -13.80),
    (109.43, -12.43),
    (109.58, -11.05),
    (109.71, -9.68),
    (109.83, -8.30),
    (109.92, -6.92),
    (110.00, -5.53),
    (110.06, -4.15),
    (110.10, -2.77),
    (110.13, -1.38),
    ]
    )

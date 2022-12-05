# -*- coding: utf-8 -*-
"""
Created on Fri Jul 04 10:51:02 2014

@author: erlean
"""

AttCoef = {'kev': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                   65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
                   95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                   108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                   120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                   132, 133, 134, 135, 136, 137, 138, 139, 140, 145, 150, 155,
                   160, 165, 170, 175, 180, 185, 190, 195, 200, 210, 220, 230,
                   240, 250, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440,
                   460, 480, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                   1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                   11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000,
                   19000, 20000],
           'Water': [0.72130, 0.64300, 0.57890, 0.52610, 0.48210, 0.44520,
                     0.41410, 0.38760, 0.36500, 0.34550, 0.32860, 0.31400,
                     0.30120, 0.28990, 0.28000, 0.27130, 0.26350, 0.25650,
                     0.25020, 0.24460, 0.23950, 0.23490, 0.23080, 0.22690,
                     0.22350, 0.22030, 0.21730, 0.21460, 0.21210, 0.20970,
                     0.20760, 0.20550, 0.20370, 0.20190, 0.20020, 0.19860,
                     0.19720, 0.19570, 0.19440, 0.19320, 0.19200, 0.19080,
                     0.18970, 0.18870, 0.18770, 0.18670, 0.18580, 0.18490,
                     0.18400, 0.18320, 0.18240, 0.18160, 0.18080, 0.18010,
                     0.17940, 0.17870, 0.17800, 0.17740, 0.17670, 0.17610,
                     0.17550, 0.17490, 0.17430, 0.17370, 0.17320, 0.17260,
                     0.17210, 0.17160, 0.17100, 0.17050, 0.17000, 0.16950,
                     0.16900, 0.16860, 0.16810, 0.16760, 0.16720, 0.16670,
                     0.16630, 0.16580, 0.16540, 0.16490, 0.16450, 0.16410,
                     0.16370, 0.16330, 0.16290, 0.16250, 0.16210, 0.16170,
                     0.16130, 0.16090, 0.16050, 0.16010, 0.15980, 0.15940,
                     0.15900, 0.15870, 0.15830, 0.15790, 0.15760, 0.15720,
                     0.15690, 0.15650, 0.15620, 0.15590, 0.15550, 0.15520,
                     0.15480, 0.15450, 0.15420, 0.15390, 0.15350, 0.15320,
                     0.15290, 0.15260, 0.15230, 0.15200, 0.15160, 0.15130,
                     0.15100, 0.14950, 0.14810, 0.14670, 0.14530, 0.14400,
                     0.14270, 0.14140, 0.14020, 0.13900, 0.13790, 0.13670,
                     0.13560, 0.13350, 0.13150, 0.12950, 0.12770, 0.12590,
                     0.12420, 0.12100, 0.11800, 0.11520, 0.11260, 0.11020,
                     0.10790, 0.10580, 0.10380, 0.10180, 0.10000, 0.09830,
                     0.09665, 0.09283, 0.08940, 0.08631, 0.08351, 0.08093,
                     0.07857, 0.07638, 0.07434, 0.07244, 0.07066, 0.04940,
                     0.03968, 0.03402, 0.03031, 0.02770, 0.02577, 0.02429,
                     0.02313, 0.02219, 0.02142, 0.02079, 0.02025, 0.01980,
                     0.01941, 0.01908, 0.01879, 0.01854, 0.01832, 0.01813],
           'Air': [0.00094, 0.00084, 0.00075, 0.00068, 0.00063, 0.00058,
                   0.00054, 0.00050, 0.00047, 0.00045, 0.00043, 0.00041,
                   0.00039, 0.00038, 0.00036, 0.00035, 0.00034, 0.00033,
                   0.00033, 0.00032, 0.00031, 0.00031, 0.00030, 0.00029,
                   0.00029, 0.00029, 0.00028, 0.00028, 0.00028, 0.00027,
                   0.00027, 0.00027, 0.00026, 0.00026, 0.00026, 0.00026,
                   0.00026, 0.00025, 0.00025, 0.00025, 0.00025, 0.00025,
                   0.00025, 0.00025, 0.00024, 0.00024, 0.00024, 0.00024,
                   0.00024, 0.00024, 0.00024, 0.00024, 0.00024, 0.00023,
                   0.00023, 0.00023, 0.00023, 0.00023, 0.00023, 0.00023,
                   0.00023, 0.00023, 0.00023, 0.00023, 0.00023, 0.00022,
                   0.00022, 0.00022, 0.00022, 0.00022, 0.00022, 0.00022,
                   0.00022, 0.00022, 0.00022, 0.00022, 0.00022, 0.00022,
                   0.00022, 0.00022, 0.00022, 0.00021, 0.00021, 0.00021,
                   0.00021, 0.00021, 0.00021, 0.00021, 0.00021, 0.00021,
                   0.00021, 0.00021, 0.00021, 0.00021, 0.00021, 0.00021,
                   0.00021, 0.00021, 0.00021, 0.00021, 0.00020, 0.00020,
                   0.00020, 0.00020, 0.00020, 0.00020, 0.00020, 0.00020,
                   0.00020, 0.00020, 0.00020, 0.00020, 0.00020, 0.00020,
                   0.00020, 0.00020, 0.00020, 0.00020, 0.00020, 0.00020,
                   0.00020, 0.00019, 0.00019, 0.00019, 0.00019, 0.00019,
                   0.00019, 0.00018, 0.00018, 0.00018, 0.00018, 0.00018,
                   0.00018, 0.00017, 0.00017, 0.00017, 0.00017, 0.00016,
                   0.00016, 0.00016, 0.00015, 0.00015, 0.00015, 0.00014,
                   0.00014, 0.00014, 0.00013, 0.00013, 0.00013, 0.00013,
                   0.00013, 0.00012, 0.00012, 0.00011, 0.00011, 0.00011,
                   0.00010, 0.00010, 0.00010, 0.00009, 0.00009, 0.00006,
                   0.00005, 0.00004, 0.00004, 0.00004, 0.00003, 0.00003,
                   0.00003, 0.00003, 0.00003, 0.00003, 0.00003, 0.00003,
                   0.00003, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002],
           'Acrylic': [0.59047, 0.53513, 0.48994, 0.45277, 0.42173, 0.39589,
                       0.37394, 0.35530, 0.33949, 0.32568, 0.31388, 0.30361,
                       0.29453, 0.28662, 0.27954, 0.27329, 0.26774, 0.26279,
                       0.25830, 0.25429, 0.25051, 0.24721, 0.24414, 0.24131,
                       0.23883, 0.23635, 0.23423, 0.23211, 0.23022, 0.22845,
                       0.22680, 0.22514, 0.22373, 0.22231, 0.22090, 0.21972,
                       0.21842, 0.21736, 0.21618, 0.21511, 0.21417, 0.21311,
                       0.21216, 0.21134, 0.21039, 0.20957, 0.20874, 0.20792,
                       0.20721, 0.20638, 0.20567, 0.20497, 0.20426, 0.20355,
                       0.20284, 0.20225, 0.20154, 0.20095, 0.20036, 0.19966,
                       0.19907, 0.19848, 0.19789, 0.19741, 0.19682, 0.19623,
                       0.19564, 0.19517, 0.19458, 0.19411, 0.19364, 0.19305,
                       0.19258, 0.19210, 0.19151, 0.19104, 0.19057, 0.19010,
                       0.18963, 0.18915, 0.18868, 0.18821, 0.18774, 0.18738,
                       0.18691, 0.18644, 0.18597, 0.18561, 0.18514, 0.18467,
                       0.18432, 0.18384, 0.18349, 0.18302, 0.18266, 0.18219,
                       0.18184, 0.18137, 0.18101, 0.18066, 0.18019, 0.17983,
                       0.17948, 0.17912, 0.17865, 0.17830, 0.17794, 0.17759,
                       0.17724, 0.17676, 0.17641, 0.17606, 0.17570, 0.17535,
                       0.17499, 0.17464, 0.17429, 0.17393, 0.17358, 0.17322,
                       0.17287, 0.17122, 0.16957, 0.16803, 0.16650, 0.16496,
                       0.16355, 0.16213, 0.16072, 0.15930, 0.15800, 0.15670,
                       0.15552, 0.15305, 0.15080, 0.14856, 0.14644, 0.14443,
                       0.14243, 0.13877, 0.13535, 0.13216, 0.12921, 0.12638,
                       0.12378, 0.12130, 0.11894, 0.11680, 0.11471, 0.11273,
                       0.11084, 0.10646, 0.10253, 0.09899, 0.09577, 0.09281,
                       0.09008, 0.08754, 0.08521, 0.08302, 0.08101, 0.05658,
                       0.04535, 0.03877, 0.03444, 0.03138, 0.02910, 0.02734,
                       0.02596, 0.02484, 0.02392, 0.02315, 0.02249, 0.02195,
                       0.02148, 0.02106, 0.02071, 0.02039, 0.02012, 0.01987],
           'Delrin': [0.82871, 0.74436, 0.67535, 0.61855, 0.57127, 0.53151,
                      0.49814, 0.46974, 0.44531, 0.42444, 0.40640, 0.39064,
                      0.37687, 0.36480, 0.35415, 0.34463, 0.33626, 0.32873,
                      0.32206, 0.31595, 0.31041, 0.30544, 0.30090, 0.29664,
                      0.29295, 0.28940, 0.28613, 0.28315, 0.28031, 0.27775,
                      0.27534, 0.27307, 0.27094, 0.26895, 0.26710, 0.26526,
                      0.26355, 0.26199, 0.26043, 0.25901, 0.25759, 0.25617,
                      0.25489, 0.25375, 0.25248, 0.25134, 0.25020, 0.24921,
                      0.24822, 0.24722, 0.24623, 0.24523, 0.24438, 0.24339,
                      0.24254, 0.24168, 0.24083, 0.24012, 0.23927, 0.23842,
                      0.23771, 0.23700, 0.23629, 0.23558, 0.23487, 0.23416,
                      0.23345, 0.23274, 0.23203, 0.23146, 0.23075, 0.23018,
                      0.22947, 0.22890, 0.22834, 0.22763, 0.22706, 0.22649,
                      0.22592, 0.22535, 0.22479, 0.22422, 0.22365, 0.22308,
                      0.22251, 0.22195, 0.22152, 0.22095, 0.22038, 0.21996,
                      0.21939, 0.21882, 0.21840, 0.21783, 0.21740, 0.21683,
                      0.21641, 0.21584, 0.21541, 0.21499, 0.21442, 0.21399,
                      0.21357, 0.21300, 0.21257, 0.21215, 0.21172, 0.21130,
                      0.21073, 0.21030, 0.20988, 0.20945, 0.20902, 0.20860,
                      0.20817, 0.20775, 0.20732, 0.20689, 0.20647, 0.20604,
                      0.20562, 0.20363, 0.20164, 0.19979, 0.19795, 0.19610,
                      0.19440, 0.19269, 0.19113, 0.18943, 0.18787, 0.18630,
                      0.18488, 0.18190, 0.17920, 0.17665, 0.17409, 0.17168,
                      0.16926, 0.16500, 0.16089, 0.15705, 0.15350, 0.15024,
                      0.14711, 0.14427, 0.14146, 0.13885, 0.13636, 0.13401,
                      0.13176, 0.12655, 0.12188, 0.11768, 0.11384, 0.11033,
                      0.10710, 0.10410, 0.10132, 0.09872, 0.09632, 0.06732,
                      0.05405, 0.04629, 0.04121, 0.03763, 0.03497, 0.03293,
                      0.03133, 0.03003, 0.02898, 0.02810, 0.02735, 0.02672,
                      0.02618, 0.02573, 0.02532, 0.02496, 0.02467, 0.02440],
           'LDPE': [0.34509, 0.32044, 0.30038, 0.28382, 0.27011, 0.25861,
                    0.24886, 0.24058, 0.23350, 0.22742, 0.22209, 0.21749,
                    0.21335, 0.20976, 0.20654, 0.20369, 0.20111, 0.19881,
                    0.19670, 0.19476, 0.19292, 0.19136, 0.18980, 0.18842,
                    0.18713, 0.18593, 0.18474, 0.18372, 0.18271, 0.18170,
                    0.18078, 0.17995, 0.17912, 0.17830, 0.17747, 0.17673,
                    0.17609, 0.17535, 0.17471, 0.17406, 0.17342, 0.17278,
                    0.17213, 0.17158, 0.17103, 0.17048, 0.16992, 0.16937,
                    0.16882, 0.16836, 0.16781, 0.16735, 0.16680, 0.16634,
                    0.16588, 0.16542, 0.16496, 0.16450, 0.16404, 0.16358,
                    0.16312, 0.16275, 0.16229, 0.16183, 0.16146, 0.16100,
                    0.16063, 0.16017, 0.15980, 0.15934, 0.15898, 0.15861,
                    0.15824, 0.15778, 0.15741, 0.15704, 0.15668, 0.15631,
                    0.15594, 0.15557, 0.15520, 0.15484, 0.15447, 0.15410,
                    0.15373, 0.15346, 0.15309, 0.15272, 0.15235, 0.15208,
                    0.15171, 0.15134, 0.15106, 0.15070, 0.15033, 0.15005,
                    0.14968, 0.14941, 0.14904, 0.14876, 0.14840, 0.14812,
                    0.14784, 0.14748, 0.14720, 0.14692, 0.14656, 0.14628,
                    0.14600, 0.14564, 0.14536, 0.14508, 0.14481, 0.14453,
                    0.14426, 0.14389, 0.14361, 0.14334, 0.14306, 0.14278,
                    0.14251, 0.14113, 0.13984, 0.13846, 0.13726, 0.13598,
                    0.13478, 0.13368, 0.13248, 0.13138, 0.13027, 0.12926,
                    0.12825, 0.12622, 0.12429, 0.12245, 0.12070, 0.11905,
                    0.11748, 0.11445, 0.11160, 0.10902, 0.10654, 0.10424,
                    0.10212, 0.10000, 0.09816, 0.09632, 0.09458, 0.09292,
                    0.09139, 0.08778, 0.08454, 0.08161, 0.07896, 0.07653,
                    0.07426, 0.07217, 0.07023, 0.06844, 0.06678, 0.04659,
                    0.03721, 0.03168, 0.02801, 0.02539, 0.02344, 0.02192,
                    0.02072, 0.01973, 0.01892, 0.01824, 0.01766, 0.01717,
                    0.01673, 0.01636, 0.01604, 0.01574, 0.01548, 0.01525],
           'PMP': [0.31133, 0.28909, 0.27100, 0.25606, 0.24369, 0.23331,
                   0.22452, 0.21705, 0.21065, 0.20518, 0.20036, 0.19621,
                   0.19248, 0.18924, 0.18634, 0.18376, 0.18144, 0.17936,
                   0.17745, 0.17571, 0.17405, 0.17264, 0.17123, 0.16998,
                   0.16882, 0.16774, 0.16666, 0.16575, 0.16484, 0.16393,
                   0.16310, 0.16235, 0.16160, 0.16085, 0.16011, 0.15944,
                   0.15886, 0.15820, 0.15762, 0.15704, 0.15646, 0.15587,
                   0.15529, 0.15480, 0.15430, 0.15380, 0.15330, 0.15280,
                   0.15231, 0.15189, 0.15139, 0.15098, 0.15048, 0.15006,
                   0.14965, 0.14923, 0.14882, 0.14840, 0.14799, 0.14757,
                   0.14716, 0.14683, 0.14641, 0.14600, 0.14567, 0.14525,
                   0.14492, 0.14450, 0.14417, 0.14376, 0.14342, 0.14309,
                   0.14276, 0.14235, 0.14201, 0.14168, 0.14135, 0.14102,
                   0.14069, 0.14035, 0.14002, 0.13969, 0.13936, 0.13903,
                   0.13869, 0.13844, 0.13811, 0.13778, 0.13745, 0.13720,
                   0.13687, 0.13654, 0.13629, 0.13595, 0.13562, 0.13537,
                   0.13504, 0.13479, 0.13446, 0.13421, 0.13388, 0.13363,
                   0.13338, 0.13305, 0.13280, 0.13255, 0.13222, 0.13197,
                   0.13172, 0.13139, 0.13114, 0.13089, 0.13064, 0.13039,
                   0.13014, 0.12981, 0.12956, 0.12931, 0.12907, 0.12882,
                   0.12857, 0.12732, 0.12616, 0.12492, 0.12384, 0.12267,
                   0.12160, 0.12060, 0.11952, 0.11852, 0.11753, 0.11662,
                   0.11570, 0.11388, 0.11213, 0.11047, 0.10890, 0.10740,
                   0.10599, 0.10325, 0.10068, 0.09836, 0.09611, 0.09404,
                   0.09213, 0.09022, 0.08856, 0.08690, 0.08532, 0.08383,
                   0.08245, 0.07919, 0.07627, 0.07363, 0.07124, 0.06904,
                   0.06700, 0.06511, 0.06336, 0.06174, 0.06025, 0.04203,
                   0.03357, 0.02859, 0.02527, 0.02291, 0.02115, 0.01978,
                   0.01869, 0.01780, 0.01707, 0.01646, 0.01594, 0.01549,
                   0.01510, 0.01476, 0.01447, 0.01420, 0.01397, 0.01376],
           'Polystyrene': [0.38738, 0.35772, 0.33351, 0.31364, 0.29716,
                           0.28335, 0.27171, 0.26183, 0.25328, 0.24596,
                           0.23968, 0.23412, 0.22928, 0.22506, 0.22124,
                           0.21785, 0.21486, 0.21208, 0.20961, 0.20734,
                           0.20528, 0.20343, 0.20167, 0.20013, 0.19858,
                           0.19725, 0.19591, 0.19467, 0.19354, 0.19240,
                           0.19137, 0.19045, 0.18952, 0.18859, 0.18777,
                           0.18695, 0.18612, 0.18540, 0.18458, 0.18396,
                           0.18324, 0.18252, 0.18190, 0.18128, 0.18066,
                           0.18004, 0.17943, 0.17881, 0.17829, 0.17778,
                           0.17716, 0.17665, 0.17613, 0.17562, 0.17510,
                           0.17459, 0.17407, 0.17366, 0.17314, 0.17263,
                           0.17222, 0.17170, 0.17129, 0.17077, 0.17036,
                           0.16995, 0.16944, 0.16902, 0.16861, 0.16820,
                           0.16779, 0.16738, 0.16696, 0.16655, 0.16614,
                           0.16573, 0.16532, 0.16490, 0.16449, 0.16418,
                           0.16377, 0.16336, 0.16295, 0.16264, 0.16223,
                           0.16181, 0.16150, 0.16109, 0.16078, 0.16037,
                           0.16006, 0.15965, 0.15934, 0.15903, 0.15862,
                           0.15831, 0.15800, 0.15759, 0.15728, 0.15697,
                           0.15656, 0.15625, 0.15594, 0.15563, 0.15532,
                           0.15502, 0.15460, 0.15429, 0.15399, 0.15368,
                           0.15337, 0.15306, 0.15275, 0.15244, 0.15213,
                           0.15182, 0.15151, 0.15120, 0.15100, 0.15069,
                           0.15038, 0.14894, 0.14750, 0.14616, 0.14482,
                           0.14348, 0.14224, 0.14101, 0.13977, 0.13864,
                           0.13751, 0.13637, 0.13524, 0.13318, 0.13122,
                           0.12927, 0.12741, 0.12566, 0.12391, 0.12072,
                           0.11783, 0.11505, 0.11248, 0.11000, 0.10774,
                           0.10558, 0.10352, 0.10165, 0.09983, 0.09810,
                           0.09645, 0.09264, 0.08922, 0.08614, 0.08334,
                           0.08076, 0.07838, 0.07617, 0.07412, 0.07222,
                           0.07048, 0.04920, 0.03937, 0.03359, 0.02976,
                           0.02705, 0.02502, 0.02346, 0.02222, 0.02122,
                           0.02038, 0.01969, 0.01910, 0.01860, 0.01816,
                           0.01779, 0.01746, 0.01717, 0.01691, 0.01669],
           'Teflon': [1.86689, 1.64506, 1.46340, 1.31328, 1.18843, 1.08367,
                      0.99511, 0.91973, 0.85536, 0.79985, 0.75190, 0.71042,
                      0.67392, 0.64217, 0.61387, 0.58903, 0.56700, 0.54734,
                      0.52985, 0.51408, 0.49982, 0.48708, 0.47542, 0.46483,
                      0.45511, 0.44647, 0.43826, 0.43092, 0.42401, 0.41774,
                      0.41191, 0.40651, 0.40154, 0.39679, 0.39226, 0.38815,
                      0.38448, 0.38081, 0.37735, 0.37411, 0.37109, 0.36828,
                      0.36547, 0.36288, 0.36050, 0.35813, 0.35575, 0.35359,
                      0.35165, 0.34970, 0.34776, 0.34603, 0.34409, 0.34258,
                      0.34085, 0.33934, 0.33782, 0.33631, 0.33480, 0.33350,
                      0.33199, 0.33070, 0.32940, 0.32832, 0.32702, 0.32573,
                      0.32465, 0.32357, 0.32249, 0.32141, 0.32033, 0.31925,
                      0.31817, 0.31730, 0.31622, 0.31514, 0.31428, 0.31342,
                      0.31234, 0.31147, 0.31061, 0.30974, 0.30888, 0.30802,
                      0.30715, 0.30629, 0.30564, 0.30478, 0.30391, 0.30326,
                      0.30240, 0.30154, 0.30089, 0.30002, 0.29938, 0.29873,
                      0.29786, 0.29722, 0.29657, 0.29570, 0.29506, 0.29441,
                      0.29376, 0.29311, 0.29225, 0.29160, 0.29095, 0.29030,
                      0.28966, 0.28901, 0.28836, 0.28793, 0.28728, 0.28663,
                      0.28598, 0.28534, 0.28469, 0.28426, 0.28361, 0.28296,
                      0.28231, 0.27950, 0.27670, 0.27410, 0.27151, 0.26892,
                      0.26654, 0.26417, 0.26179, 0.25963, 0.25747, 0.25531,
                      0.25337, 0.24926, 0.24559, 0.24192, 0.23846, 0.23501,
                      0.23198, 0.22594, 0.22032, 0.21518, 0.21034, 0.20583,
                      0.20157, 0.19755, 0.19377, 0.19019, 0.18678, 0.18354,
                      0.18045, 0.17332, 0.16695, 0.16120, 0.15595, 0.15116,
                      0.14673, 0.14260, 0.13878, 0.13524, 0.13193, 0.09240,
                      0.07463, 0.06439, 0.05774, 0.05314, 0.04974, 0.04720,
                      0.04521, 0.04363, 0.04236, 0.04130, 0.04044, 0.03972,
                      0.03912, 0.03860, 0.03817, 0.03780, 0.03748, 0.03720],
           }
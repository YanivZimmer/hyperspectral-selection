from common_utils.visualizer import Visualizer
bs_acc = {
   'EHBS': [90.79, 94.7447, 95.788, 96.279, 96.1741, 99.5424, 99.5496, 99.5583, 99.343, 99.56427015],
    'ISSC': [92.04772,92.18,96.01294776,95.67172,96.7729,96.168,99.01711,98.98,99.368,99.46,98.80411],
    'BS-Conv': [90.708, 95.3302, 95.55, 97.55, 98.01, 97.17, 98.728, 98.86, 99.0946, 98.8622177],
    'WALUDI': [89.01, 91.387, 95.86, 97.59, 97.2983, 97.3808, 97.68575766, 98.8404, 99.03652, 99.4044717],
    'WALUMI': [88.17703, 90.98, 90.074, 94.071, 95.94, 96.397, 96.13148, 98.7774, 98.6686, 98.1829, 98.142],
    'BS-FC': [90.15, 91.4, 93.7, 94.81, 94.688, 95.14, 96.17, 97.613, 98.72, 99.17207],
    'L1': [90.39, 90.371, 94.31554, 96.34761, 97.82118, 97.7656688963617, 97.6855574066264],
    'MMCA': [78.83,84.88,87.509,87.4,87.6079,90.79,94.6837,95.26,95.35,95.8762,95.46278],
    'LP': [89.66,89.314,91.73545,91.45,94.89,96.114,97.2571,96.0564254,96.24,96.65661,98.18683],
    #'RGB':[88.11],
    #'L1':[90.39,90.371,94.31554,92.7979,92.6647,98.11423]
}
bs_n_bands = {
    'EHBS': [3,4,5,6,7,13,15,17,21,25],
    'ISSC': [3,4,5,6,7,11,13,15,17,21,25],
    'BS-Conv': [3, 5, 6, 7, 11, 13, 15, 17, 21, 25],
    'WALUDI': [3, 4, 5, 6, 7, 11, 13, 17, 21, 25],
    'WALUMI': [3, 4, 5, 6, 7, 11, 13, 15, 17, 21, 25],
    'BS-FC': [3, 5, 6, 7, 11, 13, 15, 17, 21, 25],
    'L1': [3, 5, 10, 13, 15, 17,21],
    'LP': [3,4,5,6,7,11,13,15,17,21,25],
    'MMCA': [3,4,5,6,7,11,13,15,17,21,25]
    #'RGB': [3],
}
bs_only_EGHBS = {'EHBS':[90.79,
94.7447,
95.788,
96.279,
96.1741,
96.41979667,
97.6945,
97.79055796,
97.90213663,
97.604688,
99.5424,
98.75366569,
99.5496,
99.5583,
#99.10816491,
99.343,
99.56427015,
99.41354646,
#99.16911046,
99.26686217,
99.24260933,
99.31573803,
99.24242424,
99.24242424,
99.6971063945485]}
bs_n_bands_EGHBS = {'EHBS':[3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
17,
#19,
21,
25,
31,
#36,
39,
42,
50,
61,
64,
103 ]}
if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.draw_bs_methods_acc(bs_acc=bs_only_EGHBS,bs_n_bands=bs_n_bands_EGHBS)
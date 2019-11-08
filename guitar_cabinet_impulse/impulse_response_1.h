// MG impulse response from bmillier github @44.1ksps
// inserted zeros
// use with nc = 512
const float32_t guitar_cabinet_impulse [1024] = {0.21631,0,0.34689,0,0.29825,0,0.35770,0,0.18213,0,0.10291,0,-0.07526,0,-0.14740,0,-0.15997,0,-0.12085,0,-0.01587,0,0.03448,0,0.09750,0,0.07587,0,0.06842,0,0.01251,0,-0.00076,0,-0.01514,0,0.00522,0,0.02310,0,0.03232,0,0.02982,0,0.00797,0,-0.00076,0,-0.01340,0,0.00076,0,0.00967,0,0.03192,0,0.03677,0,0.03711,0,0.02206,0,0.00372,0,-0.01047,0,-0.01898,0,-0.01151,0,-0.00232,0,0.01740,0,0.02814,0,0.03976,0,0.03912,0,0.03671,0,0.02832,0,0.01993,0,0.01300,0,0.00494,0,0.00061,0,-0.00784,0,-0.01205,0,-0.02036,0,-0.02301,0,-0.02628,0,-0.02365,0,-0.01917,0,-0.01233,0,-0.00424,0,0.00049,0,0.00540,0,0.00458,0,0.00516,0,0.00177,0,0.00146,0,-0.00031,0,-0.00003,0,-0.00052,0,-0.00192,0,-0.00336,0,-0.00726,0,-0.00912,0,-0.01282,0,-0.01294,0,-0.01410,0,-0.01245,0,-0.01178,0,-0.01068,0,-0.01010,0,-0.01086,0,-0.01068,0,-0.01245,0,-0.01196,0,-0.01349,0,-0.01276,0,-0.01392,0,-0.01382,0,-0.01492,0,-0.01602,0,-0.01706,0,-0.01910,0,-0.01971,0,-0.02206,0,-0.02222,0,-0.02420,0,-0.02390,0,-0.02493,0,-0.02420,0,-0.02393,0,-0.02295,0,-0.02155,0,-0.02051,0,-0.01855,0,-0.01782,0,-0.01611,0,-0.01605,0,-0.01535,0,-0.01590,0,-0.01633,0,-0.01703,0,-0.01807,0,-0.01843,0,-0.01956,0,-0.01941,0,-0.02023,0,-0.01978,0,-0.02008,0,-0.01956,0,-0.01920,0,-0.01868,0,-0.01773,0,-0.01746,0,-0.01627,0,-0.01627,0,-0.01520,0,-0.01535,0,-0.01462,0,-0.01474,0,-0.01447,0,-0.01447,0,-0.01474,0,-0.01459,0,-0.01523,0,-0.01483,0,-0.01547,0,-0.01492,0,-0.01523,0,-0.01465,0,-0.01456,0,-0.01416,0,-0.01373,0,-0.01358,0,-0.01285,0,-0.01288,0,-0.01202,0,-0.01221,0,-0.01154,0,-0.01178,0,-0.01154,0,-0.01175,0,-0.01196,0,-0.01199,0,-0.01254,0,-0.01233,0,-0.01291,0,-0.01257,0,-0.01303,0,-0.01263,0,-0.01282,0,-0.01251,0,-0.01233,0,-0.01218,0,-0.01172,0,-0.01172,0,-0.01108,0,-0.01117,0,-0.01056,0,-0.01065,0,-0.01022,0,-0.01016,0,-0.00998,0,-0.00967,0,-0.00964,0,-0.00909,0,-0.00912,0,-0.00842,0,-0.00842,0,-0.00772,0,-0.00760,0,-0.00705,0,-0.00671,0,-0.00635,0,-0.00580,0,-0.00565,0,-0.00494,0,-0.00491,0,-0.00421,0,-0.00418,0,-0.00360,0,-0.00348,0,-0.00308,0,-0.00284,0,-0.00269,0,-0.00232,0,-0.00235,0,-0.00192,0,-0.00198,0,-0.00159,0,-0.00159,0,-0.00125,0,-0.00116,0,-0.00095,0,-0.00067,0,-0.00064,0,-0.00027,0,-0.00031,0,0.00003,0,0.00000,0,0.00027,0,0.00027,0,0.00049,0,0.00061,0,0.00067,0,0.00095,0,0.00095,0,0.00131,0,0.00125,0,0.00162,0,0.00162,0,0.00192,0,0.00198,0,0.00217,0,0.00238,0,0.00244,0,0.00278,0,0.00275,0,0.00311,0,0.00308,0,0.00342,0,0.00345,0,0.00369,0,0.00378,0,0.00391,0,0.00412,0,0.00409,0,0.00436,0,0.00427,0,0.00455,0,0.00446,0,0.00467,0,0.00467,0,0.00479,0,0.00488,0,0.00491,0,0.00510,0,0.00507,0,0.00531,0,0.00522,0,0.00546,0,0.00543,0,0.00562,0,0.00562,0,0.00568,0,0.00577,0,0.00574,0,0.00589,0,0.00583,0,0.00598,0,0.00589,0,0.00604,0,0.00598,0,0.00607,0,0.00610,0,0.00607,0,0.00620,0,0.00610,0,0.00623,0,0.00610,0,0.00626,0,0.00613,0,0.00623,0,0.00613,0,0.00613,0,0.00613,0,0.00607,0,0.00610,0,0.00598,0,0.00607,0,0.00592,0,0.00601,0,0.00589,0,0.00589,0,0.00583,0,0.00580,0,0.00580,0,0.00571,0,0.00577,0,0.00562,0,0.00568,0,0.00555,0,0.00555,0,0.00546,0,0.00543,0,0.00537,0,0.00528,0,0.00528,0,0.00513,0,0.00516,0,0.00500,0,0.00500,0,0.00488,0,0.00488,0,0.00479,0,0.00473,0,0.00470,0,0.00458,0,0.00458,0,0.00443,0,0.00443,0,0.00430,0,0.00427,0,0.00418,0,0.00412,0,0.00403,0,0.00394,0,0.00388,0,0.00375,0,0.00375,0,0.00360,0,0.00360,0,0.00345,0,0.00342,0,0.00333,0,0.00323,0,0.00317,0,0.00308,0,0.00305,0,0.00293,0,0.00290,0,0.00278,0,0.00275,0,0.00262,0,0.00256,0,0.00247,0,0.00238,0,0.00232,0,0.00220,0,0.00217,0,0.00204,0,0.00201,0,0.00189,0,0.00183,0,0.00174,0,0.00165,0,0.00159,0,0.00150,0,0.00143,0,0.00134,0,0.00128,0,0.00116,0,0.00113,0,0.00101,0,0.00095,0,0.00089,0,0.00079,0,0.00073,0,0.00064,0,0.00061,0,0.00049,0,0.00046,0,0.00037,0,0.00034,0,0.00024,0,0.00018,0,0.00012,0,0.00006,0,0.00000,0,-0.00003,0,-0.00006,0,-0.00015,0,-0.00018,0,-0.00024,0,-0.00027,0,-0.00034,0,-0.00037,0,-0.00043,0,-0.00046,0,-0.00049,0,-0.00055,0,-0.00058,0,-0.00064,0,-0.00064,0,-0.00070,0,-0.00070,0,-0.00076,0,-0.00079,0,-0.00082,0,-0.00085,0,-0.00085,0,-0.00089,0,-0.00089,0,-0.00095,0,-0.00095,0,-0.00098,0,-0.00098,0,-0.00101,0,-0.00104,0,-0.00104,0,-0.00104,0,-0.00104,0,-0.00107,0,-0.00107,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00113,0,-0.00110,0,-0.00113,0,-0.00110,0,-0.00113,0,-0.00113,0,-0.00113,0,-0.00113,0,-0.00113,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00110,0,-0.00107,0,-0.00107,0,-0.00107,0,-0.00104,0,-0.00104,0,-0.00104,0,-0.00101,0,-0.00101,0,-0.00101,0,-0.00098,0,-0.00098,0,-0.00095,0,-0.00095,0,-0.00095,0,-0.00092,0,-0.00092,0,-0.00089,0,-0.00089,0,-0.00085,0,-0.00085,0,-0.00082,0,-0.00082,0,-0.00082,0,-0.00079,0,-0.00079,0,-0.00076,0,-0.00076,0,-0.00073,0,-0.00073,0,-0.00070,0,-0.00070,0,-0.00067,0,-0.00067,0,-0.00064,0,-0.00064,0,-0.00061,0,-0.00061,0,-0.00058,0,-0.00058,0,-0.00055,0,-0.00055,0,-0.00052,0,-0.00052,0,-0.00049,0,-0.00049,0,-0.00049,0,-0.00046,0,-0.00046,0,-0.00043,0,-0.00043,0,-0.00040,0,-0.00040,0,-0.00040,0,-0.00037,0,-0.00037,0,-0.00034,0,-0.00034,0,-0.00034,0,-0.00031,0,-0.00031,0,-0.00027,0,-0.00027,0,-0.00027,0,-0.00024,0,-0.00024,0,-0.00024,0,-0.00021,0,-0.00021,0,-0.00021,0,-0.00018,0,-0.00018,0,-0.00018,0,-0.00015,0,-0.00015,0,-0.00015,0,-0.00012,0,-0.00012,0,-0.00012,0,-0.00012,0,-0.00009,0,-0.00009,0,-0.00009,0,-0.00006,0,-0.00006,0,-0.00006,0,-0.00006,0,-0.00003,0,-0.00003,0,-0.00003,0,-0.00003,0,-0.00003,0,0.00000,0,0.00000,0};

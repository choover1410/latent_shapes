import numpy as np

y_points = [0.41467085, 0.41483114, 0.41838077, 0.420449, 0.42491956, 0.43108353,
    0.43998984, 0.44097028, 0.45290862, 0.46464842, 0.47677692, 0.50906301,
    0.52226878, 0.53532645, 0.54856748, 0.56186702, 0.57496355, 0.58844849,
    0.60203725, 0.61568138, 0.62954352, 0.64341272, 0.65738954, 0.67145767,
    0.68550832, 0.69967037, 0.71384532, 0.72803857, 0.74227122, 0.75653205,
    0.77079719, 0.78508297, 0.79939678, 0.8136982, 0.82804055, 0.84237627,
    0.85670621, 0.87108935, 0.88543467, 0.89980128, 0.91419061, 0.9285569,
    0.94294632, 0.95733567, 0.97172386, 0.9945424, 0.16053395, 0.06926921,
    0.02503454, 0.0]


x_points = np.array([
(8.45601499e-02, 0.00000000e+00),
(9.21650276e-02, 2.60924774e-03),
(1.01093552e-01, 6.33837833e-03),
(1.10291720e-01, 1.08514568e-02),
(1.18268469e-01, 1.54615552e-02),
(1.23819408e-01, 1.89924907e-02),
(1.29712313e-01, 2.43937242e-02),
(1.35370431e-01, 3.05783188e-02),
(1.39650042e-01, 3.57897221e-02),
(1.44991817e-01, 4.31177233e-02),
(1.47893062e-01, 4.71335537e-02),
(1.47395683e-01, 4.97118021e-02),
(1.43535164e-01, 5.49274587e-02),
(1.39484250e-01, 6.00934228e-02),
(1.35512543e-01, 6.46713610e-02),
(1.32635523e-01, 6.79809801e-02),
(1.31993053e-01, 6.95188600e-02),
(1.29178573e-01, 7.53636860e-02),
(1.26414443e-01, 8.07550961e-02),
(1.23987398e-01, 8.49559909e-02),
(1.21241649e-01, 8.92904212e-02),
(1.19444379e-01, 9.17314197e-02),
(1.16896372e-01, 9.48977513e-02),
(1.13129887e-01, 9.93863079e-02),
(1.08485616e-01, 1.04408830e-01),
(1.07046540e-01, 1.05958591e-01),
(1.04427468e-01, 1.08209924e-01),
(9.95930341e-02, 1.12160178e-01),
(9.46314063e-02, 1.15844575e-01),
(8.98558815e-02, 1.19056796e-01),
(8.71228801e-02, 1.21068667e-01),
(8.57943623e-02, 1.22828646e-01),
(8.15771721e-02, 1.27946688e-01),
(7.66638769e-02, 1.33231643e-01),
(7.28598139e-02, 1.37178570e-01),
(7.02388516e-02, 1.38655412e-01),
(6.58649055e-02, 1.36752646e-01),
(5.92602075e-02, 1.33434992e-01),
(5.14504179e-02, 1.28948623e-01),
(4.38724814e-02, 1.23918937e-01),
(3.93915212e-02, 1.20515317e-01),
(3.36295044e-02, 1.14280143e-01),
(2.73954278e-02, 1.06436190e-01),
(2.15434643e-02, 9.81086579e-02),
(1.57981415e-02, 8.84295524e-02),
(1.37938687e-02, 8.42118169e-02),
(1.21575528e-02, 8.36361328e-02),
(7.99953715e-03, 8.42320906e-02),
(2.65236493e-03, 8.44972392e-02),
(3.19017213e-04, 8.44957192e-02),
(-5.85283679e-04, 8.61820693e-02),
(-2.32829167e-03, 9.14228508e-02),
(-5.96289033e-03, 1.00289714e-01),
(-1.04702910e-02, 1.09533163e-01),
(-1.52233304e-02, 1.17940869e-01),
(-1.97769544e-02, 1.24814493e-01),
(-2.47875849e-02, 1.30039258e-01),
(-2.97236156e-02, 1.34633559e-01),
(-3.58826308e-02, 1.39695884e-01),
(-4.43600060e-02, 1.45865007e-01),
(-4.73832621e-02, 1.48041922e-01),
(-5.06100762e-02, 1.46854574e-01),
(-5.51366252e-02, 1.43378036e-01),
(-6.05948195e-02, 1.39099652e-01),
(-6.56306171e-02, 1.34612038e-01),
(-6.84822928e-02, 1.32358895e-01),
(-7.05523018e-02, 1.31484078e-01),
(-7.48966020e-02, 1.29444436e-01),
(-8.02151037e-02, 1.26679652e-01),
(-8.53764937e-02, 1.23737276e-01),
(-8.93489016e-02, 1.21238911e-01),
(-9.10894712e-02, 1.19915313e-01),
(-9.51247135e-02, 1.16706947e-01),
(-9.94820301e-02, 1.13042955e-01),
(-1.03400230e-01, 1.09472531e-01),
(-1.06858966e-01, 1.05993747e-01),
(-1.08926659e-01, 1.03647746e-01),
(-1.11762800e-01, 1.00054557e-01),
(-1.15667945e-01, 9.49007069e-02),
(-1.19012933e-01, 8.99795797e-02),
(-1.20114235e-01, 8.81711201e-02),
(-1.22753603e-01, 8.58300709e-02),
(-1.27917739e-01, 8.15960689e-02),
(-1.32766919e-01, 7.71074082e-02),
(-1.37009372e-01, 7.30410978e-02),
(-1.38896841e-01, 7.02158393e-02),
(-1.37414538e-01, 6.72117714e-02),
(-1.32409483e-01, 5.73774333e-02),
(-1.28397407e-01, 5.05441199e-02),
(-1.24610049e-01, 4.48763273e-02),
(-1.20047626e-01, 3.88841844e-02),
(-1.13445472e-01, 3.29603419e-02),
(-1.06840688e-01, 2.76862517e-02),
(-9.92499894e-02, 2.23471564e-02),
(-9.06818566e-02, 1.69187880e-02),
(-8.49687819e-02, 1.42110756e-02),
(-8.35320685e-02, 1.11992920e-02),
(-8.39458130e-02, 1.01146062e-02),
(-8.40103866e-02, 1.12796796e-02),
(-8.44800596e-02, 2.72508925e-03),
(-8.51915000e-02, -5.17531477e-04),
(-8.95308644e-02, -1.69870179e-03),
(-9.98628150e-02, -5.73655875e-03),
(-1.09160937e-01, -1.02765482e-02),
(-1.18055852e-01, -1.52900668e-02),
(-1.25064497e-01, -1.99485008e-02),
(-1.29076028e-01, -2.37352959e-02),
(-1.35169321e-01, -3.03960286e-02),
(-1.39903793e-01, -3.61193855e-02),
(-1.43726965e-01, -4.13319034e-02),
(-1.47719685e-01, -4.72674016e-02),
(-1.47649119e-01, -4.94034765e-02),
(-1.44061246e-01, -5.42524355e-02),
(-1.38855203e-01, -6.08855858e-02),
(-1.34732205e-01, -6.55587118e-02),
(-1.33182956e-01, -6.71619240e-02),
(-1.31606219e-01, -7.01980710e-02),
(-1.29313172e-01, -7.51694072e-02),
(-1.26478739e-01, -8.05923805e-02),
(-1.23118804e-01, -8.63894803e-02),
(-1.21507431e-01, -8.91062274e-02),
(-1.20267124e-01, -9.06412479e-02),
(-1.15312839e-01, -9.68553383e-02),
(-1.13093577e-01, -9.94676590e-02),
(-1.10211672e-01, -1.02587256e-01),
(-1.06099583e-01, -1.06799299e-01),
(-1.02428620e-01, -1.09864153e-01),
(-9.99695161e-02, -1.11888282e-01),
(-9.59379781e-02, -1.14874387e-01),
(-9.06151440e-02, -1.18637775e-01),
(-8.80715773e-02, -1.20222763e-01),
(-8.58101450e-02, -1.22827444e-01),
(-8.21326693e-02, -1.27252000e-01),
(-7.69877514e-02, -1.32925837e-01),
(-7.28239574e-02, -1.37130355e-01),
(-7.10918011e-02, -1.38842131e-01),
(-6.59548178e-02, -1.36835199e-01),
(-5.63965203e-02, -1.31877928e-01),
(-5.18391958e-02, -1.29195419e-01),
(-4.47896593e-02, -1.24572639e-01),
(-3.83604813e-02, -1.19578721e-01),
(-3.40579859e-02, -1.14751293e-01),
(-2.78249548e-02, -1.07040709e-01),
(-2.17992331e-02, -9.84273541e-02),
(-1.68497284e-02, -9.03562638e-02),
(-1.36415111e-02, -8.44311401e-02),
(-1.29173040e-02, -8.35965546e-02),
(-1.13728808e-02, -8.37715893e-02),
(-7.23314736e-03, -8.41684135e-02),
(-6.11615844e-04, -8.44924837e-02),
(-2.27011003e-04, -8.40397052e-02),
(1.85466215e-03, -9.00752838e-02),
(5.71193816e-03, -9.97983172e-02),
(1.02694483e-02, -1.09143464e-01),
(1.54951546e-02, -1.18376613e-01),
(1.92495535e-02, -1.24210767e-01),
(2.36197494e-02, -1.28973633e-01),
(3.09472807e-02, -1.35649207e-01),
(3.51882108e-02, -1.39202441e-01),
(4.17107418e-02, -1.44015968e-01),
(4.78120667e-02, -1.48094138e-01),
(4.91154633e-02, -1.47691858e-01),
(5.50697175e-02, -1.43463825e-01),
(6.08793899e-02, -1.38828887e-01),
(6.50983975e-02, -1.35169615e-01),
(6.73968231e-02, -1.33037574e-01),
(6.98450212e-02, -1.31807332e-01),
(7.44220429e-02, -1.29653975e-01),
(8.05991608e-02, -1.26481033e-01),
(8.62628653e-02, -1.23202216e-01),
(8.80876980e-02, -1.22149303e-01),
(9.15216420e-02, -1.19547446e-01),
(9.64700256e-02, -1.15644021e-01),
(9.88587808e-02, -1.13612432e-01),
(1.03897352e-01, -1.08957521e-01),
(1.06877937e-01, -1.06101810e-01),
(1.07932403e-01, -1.04749528e-01),
(1.11815678e-01, -1.00052629e-01),
(1.16289716e-01, -9.39623501e-02),
(1.19304049e-01, -8.95327061e-02),
(1.20898155e-01, -8.73545780e-02),
(1.23827575e-01, -8.50246593e-02),
(1.27291830e-01, -8.21316374e-02),
(1.32320124e-01, -7.75384809e-02),
(1.37797021e-01, -7.20658978e-02),
(1.38725981e-01, -7.05109326e-02),
(1.35752639e-01, -6.37641965e-02),
(1.32231132e-01, -5.69961993e-02),
(1.29451911e-01, -5.22905453e-02),
(1.23997593e-01, -4.39415085e-02),
(1.20168203e-01, -3.89854675e-02),
(1.14896622e-01, -3.41649375e-02),
(1.06559162e-01, -2.74752918e-02),
(9.85694800e-02, -2.18655484e-02),
(9.00469136e-02, -1.67170814e-02),
(8.43745663e-02, -1.36501467e-02),
(8.35893863e-02, -1.32456490e-02),
(8.37464614e-02, -1.17508123e-02),
(8.41664767e-02, -8.06798222e-03),
(8.44892465e-02, -3.68686404e-03),
(8.45601238e-02, -6.64133796e-05),
])
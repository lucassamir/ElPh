%mem=40GB
%nprocshared=24
#b3lyp/6-31+g* scf=tight nosymm punch=mo iop(3/33=1)

Gaussian input prepared by ASE

0 1
C                 1.7960000000       10.8247500000       20.9070381000
C                 1.7960000000       10.8247500000       19.4384619000
C                 2.8218752000       11.5319670000       21.5902219000
C                 0.7701248000       10.1175330000       21.5902219000
C                 0.7701248000       10.1175330000       18.7552781000
C                 2.8218752000       11.5319670000       18.7552781000
C                 3.8980384000       12.0876375000       20.8935896000
C                -0.3060384000        9.5618625000       20.8935896000
C                -0.3060384000        9.5618625000       19.4519104000
C                 3.8980384000       12.0876375000       19.4519104000
C                 4.9785120000       12.7529988000       21.5579455000
C                -1.3865120000        8.8965012000       21.5579455000
C                -1.3865120000        8.8965012000       18.7875545000
C                 4.9785120000       12.7529988000       18.7875545000
C                 5.9849904000       13.3433085000       20.8801411000
C                -2.3929904000        8.3061915000       20.8801411000
C                -2.3929904000        8.3061915000       19.4653589000
C                 5.9849904000       13.3433085000       19.4653589000
C                 2.7450064000       11.8725858000       23.0561084000
C                 0.8469936000        9.7769142000       23.0561084000
C                 0.8469936000        9.7769142000       17.2893916000
C                 2.7450064000       11.8725858000       17.2893916000
C                 1.9554848000       12.9377412000       23.4514943000
C                 1.6365152000        8.7117588000       23.4514943000
C                 1.6365152000        8.7117588000       16.8940057000
C                 1.9554848000       12.9377412000       16.8940057000
C                 1.9562032000       13.3707312000       24.7748267000
C                 1.6357968000        8.2787688000       24.7748267000
C                 1.6357968000        8.2787688000       15.5706733000
C                 1.9562032000       13.3707312000       15.5706733000
C                 2.7658400000       12.7515555000       25.6973938000
C                 0.8261600000        8.8979445000       25.6973938000
C                 0.8261600000        8.8979445000       14.6481062000
C                 2.7658400000       12.7515555000       14.6481062000
C                 3.5567984000       11.7051630000       25.3100770000
C                 0.0352016000        9.9443370000       25.3100770000
C                 0.0352016000        9.9443370000       15.0354230000
C                 3.5567984000       11.7051630000       15.0354230000
C                 3.5510512000       11.2591833000       24.0001931000
C                 0.0409488000       10.3903167000       24.0001931000
C                 0.0409488000       10.3903167000       16.3453069000
C                 3.5510512000       11.2591833000       16.3453069000
H                 4.9354080000       12.8165040000       22.5396860000
H                -1.3434080000        8.8329960000       22.5396860000
H                -1.3434080000        8.8329960000       17.8058140000
H                 4.9354080000       12.8165040000       17.8058140000
H                 6.7242240000       13.8123810000       21.3831150000
H                -3.1322240000        7.8371190000       21.3831150000
H                -3.1322240000        7.8371190000       18.9623850000
H                 6.7242240000       13.8123810000       18.9623850000
H                 1.3793280000       13.3938240000       22.7817590000
H                 2.2126720000        8.2556760000       22.7817590000
H                 2.2126720000        8.2556760000       17.5637410000
H                 1.3793280000       13.3938240000       17.5637410000
H                 1.3649600000       14.1732060000       25.0142100000
H                 2.2270400000        7.4762940000       25.0142100000
H                 2.2270400000        7.4762940000       15.3312900000
H                 1.3649600000       14.1732060000       15.3312900000
H                 2.7802080000       13.0762980000       26.6549270000
H                 0.8117920000        8.5732020000       26.6549270000
H                 0.8117920000        8.5732020000       13.6905730000
H                 2.7802080000       13.0762980000       13.6905730000
H                 4.1739040000       11.2433070000       25.9556050000
H                -0.5819040000       10.4061930000       25.9556050000
H                -0.5819040000       10.4061930000       14.3898950000
H                 4.1739040000       11.2433070000       14.3898950000
H                 4.1164320000       10.4494920000       23.7231540000
H                -0.5244320000       11.2000080000       23.7231540000
H                -0.5244320000       11.2000080000       16.6223460000
H                 4.1164320000       10.4494920000       16.6223460000
C                 5.3880000000        3.6082500000       20.9070381000
C                 5.3880000000        3.6082500000       19.4384619000
C                 4.3621248000        4.3154670000       21.5902219000
C                 6.4138752000        2.9010330000       21.5902219000
C                 6.4138752000        2.9010330000       18.7552781000
C                 4.3621248000        4.3154670000       18.7552781000
C                 3.2859616000        4.8711375000       20.8935896000
C                 7.4900384000        2.3453625000       20.8935896000
C                 7.4900384000        2.3453625000       19.4519104000
C                 3.2859616000        4.8711375000       19.4519104000
C                 2.2054880000        5.5364988000       21.5579455000
C                 8.5705120000        1.6800012000       21.5579455000
C                 8.5705120000        1.6800012000       18.7875545000
C                 2.2054880000        5.5364988000       18.7875545000
C                 1.1990096000        6.1268085000       20.8801411000
C                 9.5769904000        1.0896915000       20.8801411000
C                 9.5769904000        1.0896915000       19.4653589000
C                 1.1990096000        6.1268085000       19.4653589000
C                 4.4389936000        4.6560858000       23.0561084000
C                 6.3370064000        2.5604142000       23.0561084000
C                 6.3370064000        2.5604142000       17.2893916000
C                 4.4389936000        4.6560858000       17.2893916000
C                 5.2285152000        5.7212412000       23.4514943000
C                 5.5474848000        1.4952588000       23.4514943000
C                 5.5474848000        1.4952588000       16.8940057000
C                 5.2285152000        5.7212412000       16.8940057000
C                 5.2277968000        6.1542312000       24.7748267000
C                 5.5482032000        1.0622688000       24.7748267000
C                 5.5482032000        1.0622688000       15.5706733000
C                 5.2277968000        6.1542312000       15.5706733000
C                 4.4181600000        5.5350555000       25.6973938000
C                 6.3578400000        1.6814445000       25.6973938000
C                 6.3578400000        1.6814445000       14.6481062000
C                 4.4181600000        5.5350555000       14.6481062000
C                 3.6272016000        4.4886630000       25.3100770000
C                 7.1487984000        2.7278370000       25.3100770000
C                 7.1487984000        2.7278370000       15.0354230000
C                 3.6272016000        4.4886630000       15.0354230000
C                 3.6329488000        4.0426833000       24.0001931000
C                 7.1430512000        3.1738167000       24.0001931000
C                 7.1430512000        3.1738167000       16.3453069000
C                 3.6329488000        4.0426833000       16.3453069000
H                 2.2485920000        5.6000040000       22.5396860000
H                 8.5274080000        1.6164960000       22.5396860000
H                 8.5274080000        1.6164960000       17.8058140000
H                 2.2485920000        5.6000040000       17.8058140000
H                 0.4597760000        6.5958810000       21.3831150000
H                10.3162240000        0.6206190000       21.3831150000
H                10.3162240000        0.6206190000       18.9623850000
H                 0.4597760000        6.5958810000       18.9623850000
H                 5.8046720000        6.1773240000       22.7817590000
H                 4.9713280000        1.0391760000       22.7817590000
H                 4.9713280000        1.0391760000       17.5637410000
H                 5.8046720000        6.1773240000       17.5637410000
H                 5.8190400000        6.9567060000       25.0142100000
H                 4.9569600000        0.2597940000       25.0142100000
H                 4.9569600000        0.2597940000       15.3312900000
H                 5.8190400000        6.9567060000       15.3312900000
H                 4.4037920000        5.8597980000       26.6549270000
H                 6.3722080000        1.3567020000       26.6549270000
H                 6.3722080000        1.3567020000       13.6905730000
H                 4.4037920000        5.8597980000       13.6905730000
H                 3.0100960000        4.0268070000       25.9556050000
H                 7.7659040000        3.1896930000       25.9556050000
H                 7.7659040000        3.1896930000       14.3898950000
H                 3.0100960000        4.0268070000       14.3898950000
H                 3.0675680000        3.2329920000       23.7231540000
H                 7.7084320000        3.9835080000       23.7231540000
H                 7.7084320000        3.9835080000       16.6223460000
H                 3.0675680000        3.2329920000       16.6223460000



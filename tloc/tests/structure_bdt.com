%mem=24GB
%nprocshared=12
#b3lyp/6-31+g* scf=tight nosymm punch=mo iop(3/33=1)

Gaussian input prepared by ASE

0 1
Si               10.2855105976        6.9661745914        3.2197193751
Si                8.6474257710        8.0757418971       14.8498884591
C                 9.5872804016        7.5274555477        7.5937412680
C                 9.6968228081        6.3707504100        8.3636895639
C                 9.9817702467        4.0786438516        9.3219182920
C                 9.7667909229        5.0458596797       10.2532245124
C                 9.6395501311        6.3664655319        9.7806636536
C                 9.4408348562        7.5762239811       10.4893278201
C                 9.3351453064        8.7295422561        9.7098149491
C                 9.0153445463       11.0199704077        8.7267891744
C                 9.2334262259       10.0445276887        7.8073501120
C                 9.3831306164        8.7297547667        8.2963832947
C                10.1093005470        2.6518668170        9.5397780583
C                10.3619441733        0.3646718296       10.5068628745
C                10.4921901024       -0.8110762430       11.2330620954
C                10.7141691805       -1.9928837730       10.5281174858
C                10.8922409429       -3.2927638719       11.0842798160
C                11.1052596344       -4.2328529598       10.1242798703
C                10.7564897544       -1.9845265273        9.1076009610
C                10.6312667526       -0.7992733226        8.3831729577
C                10.4493354945        0.3825342074        9.0881175673
C                10.2925021187        1.7092786600        8.5702135375
C                 8.8656076046       12.4450690354        8.4841323616
C                 8.5969434732       14.7040401671        7.4373427529
C                 8.4761783044       15.8444708590        6.6615494388
C                 8.2557639422       17.0487250009        7.3346121314
C                 8.0998797476       18.3204960706        6.7341693609
C                 7.8757026893       19.3171831146        7.6534313015
C                 8.0793207311       17.0811002543        8.7498150034
C                 8.2470239173       15.9213412883        9.5061249237
C                 8.4909168250       14.7223167273        8.8578592777
C                 8.6421927290       13.4125772613        9.4152614602
C                 9.7284857525        7.3825610764        6.1811952224
C                 9.9030140031        7.2275721778        5.0013871711
C                11.1410328286        8.4990921289        2.5815496695
C                10.2856991087        9.7746446811        2.6957932055
C                10.9181169165       11.0186538855        2.0595718393
C                12.1027958997       11.6050496359        2.8180072207
C                11.7213222515       12.3444725427        4.0853134221
C                12.9176322842       12.9302737921        4.8109812778
C                 8.6463547881        6.7067218699        2.3721917478
C                 8.6631091552        6.6905133323        0.8446936792
C                 7.3354187962        6.2605343866        0.2221106885
C                 6.1248050002        6.9982323130        0.7710110265
C                 4.8202496129        6.6362133161        0.1016678909
C                 3.6080497622        7.2288448596        0.7543615810
C                11.3286698090        5.4118407286        3.1285016681
C                10.5266854870        4.1440325696        3.4944352267
C                11.3201960046        2.8984669818        3.7737562442
C                10.4493115245        1.6512760140        3.9597340934
C                11.1651883719        0.4012475731        4.3357635925
C                10.2710946007       -0.7903692591        4.4445163538
C                 9.3305237783        7.7129215979       11.9093129797
C                 9.1863240368        7.8450292710       13.0992169714
C                 8.1987940567        9.8751403147       15.0199253499
C                 7.1267084048       10.3743164856       14.0438073239
C                 6.8578277195       11.8811220410       14.1171357331
C                 8.0583437819       12.7426016657       13.7830840914
C                 7.7550206763       14.2402872806       13.7434088169
C                 8.9738971778       15.0888042123       13.4775490534
C                10.0097528946        7.6089882208       16.0479400519
C                11.2019426938        8.5644775359       16.0977112668
C                12.3664190423        8.0473803511       16.9379769019
C                13.5480121673        9.0126698429       16.9983754225
C                14.7723468694        8.4202118627       17.6508919908
C                15.9253264864        9.4188205413       17.7497259335
C                 7.2073172841        6.9160546629       15.1024640906
C                 7.5963097882        5.4352955088       14.9136522932
C                 6.4324338227        4.4833246090       15.0728847565
C                 6.8611653161        3.0034850746       14.8144641069
C                 7.0924536484        2.6524776415       13.3709217531
C                 7.4484082976        1.1708404118       13.1796302510
S                 9.9554216210        4.7770773299        7.7216231796
S                 9.0949664253       10.3449260885       10.3394828101
S                10.1090218689        1.9658403652       11.1359993703
S                11.0319856536       -3.5854212315        8.5218592967
S                 8.8833612840       13.0837111729        6.8624055160
S                 7.8886059823       18.7144078357        9.2751581471
H                 9.7187909325        4.8445395805       11.1804569323
H                 9.2329070063       10.2279337946        6.8751582828
H                10.4296124817       -0.8111119134       12.1806635178
H                10.8633463321       -3.4801813904       12.0141690623
H                11.2785568230       -5.1476425473       10.3138001548
H                10.6671257842       -0.8016404361        7.4338003176
H                10.3132055444        1.9126667439        7.6428039959
H                 8.5418452341       15.8089947881        5.7139480164
H                 8.1445149090       18.4698930606        5.7971952441
H                 7.7312750486       20.2294550918        7.4267154472
H                 8.1941569926       15.9556689688       10.4554975637
H                 8.5921397008       13.2362735036       10.3474532894
H                11.9818715676        8.6311982767        3.0890035154
H                11.3814818928        8.3582707437        1.6312914207
H                 9.4095450637        9.6090965163        2.2653873258
H                10.1176295278        9.9587718368        3.6540219336
H                11.2159932798       10.7852510819        1.1442065774
H                10.2212516840       11.7174693290        1.9766788550
H                12.7251571794       10.8718743203        3.0535791631
H                12.5870474082       12.2277605970        2.2193356679
H                11.2495715098       11.7201449995        4.6919554542
H                11.0943371665       13.0754040794        3.8559407414
H                13.5208501635       12.2079189447        5.0816233289
H                12.6116140892       13.4163073420        5.6059037420
H                13.3921835210       13.5465617212        4.2137266990
H                 8.2693736777        5.8461238625        2.6851658998
H                 8.0296391815        7.4230483555        2.6656825061
H                 9.3736609535        6.0731877554        0.5384501541
H                 8.8911608837        7.5962174398        0.5171955427
H                 7.3798824812        6.4066187289       -0.7563099203
H                 7.2117636512        5.2904189331        0.3737269161
H                 6.0482799666        6.8056037421        1.7393356950
H                 6.2734333613        7.9715376218        0.6695202573
H                 4.8534307469        6.9375987938       -0.8413283657
H                 4.7296993897        5.6506776950        0.0974169687
H                 3.5135342568        6.8643836524        1.6596309025
H                 2.8123047007        7.0056979210        0.2284870720
H                 3.7055522629        8.2032384355        0.8023615782
H                11.6873388232        5.3158153618        2.2104795798
H                12.0963154379        5.4997780045        3.7478964670
H                 9.9794824818        4.3459112347        4.2952027091
H                 9.9014780690        3.9505477285        2.7524721691
H                11.8590800267        3.0379129959        4.5927672680
H                11.9472520192        2.7432279356        3.0234684637
H                 9.7759171369        1.8448421410        4.6583023196
H                 9.9605127022        1.4867659631        3.1155717795
H                11.6175611066        0.5411988614        5.2038373441
H                11.8629833137        0.2162316408        3.6593355864
H                 9.5924402330       -0.6273670771        5.1329886396
H                10.8029797909       -1.5760781041        4.6901842366
H                 9.8321330148       -0.9488891269        3.5831732291
H                 7.8802962026       10.0335011523       15.9445009434
H                 9.0173974668       10.4167191484       14.8906264642
H                 7.4071811009       10.1470425517       13.1211800698
H                 6.2815465671        9.8927268571       14.2281910773
H                 6.1228612285       12.1034301792       13.4913645507
H                 6.5503813694       12.1051345515       15.0305526556
H                 8.4134023738       12.4663126807       12.9015490859
H                 8.7654037196       12.5784214607       14.4566781493
H                 7.0841694911       14.4127094011       13.0379328421
H                 7.3593448884       14.5109314945       14.6090028639
H                 9.6209487364       14.9674518554       14.2033940308
H                 8.7114021045       16.0315323036       13.4276007167
H                 9.3811904076       14.8182589307       12.6287815737
H                 9.6197794159        7.5489156374       16.9558661998
H                10.3418112267        6.7068460893       15.8063459696
H                11.5208740936        8.7255032914       15.1740212821
H                10.9016500962        9.4325663352       16.4687813565
H                12.6754899862        7.1853267831       16.5608846724
H                12.0451616826        7.8758883499       17.8591871820
H                13.7787583488        9.2939558440       16.0773422643
H                13.2754991506        9.8213046970       17.5014012243
H                15.0700696697        7.6346119452       17.1276743082
H                14.5361788766        8.1082779962       18.5605893563
H                16.1430386893        9.7563590741       16.8566780136
H                16.7111760613        8.9732805611       18.1301834766
H                15.6616101775       10.1668460142       18.3267886315
H                 6.8477200523        7.0418666087       16.0153496479
H                 6.4900475463        7.1463794501       14.4602205845
H                 7.9839869839        5.3180196282       14.0103313110
H                 8.2945650370        5.2006624872       15.5743164625
H                 5.7160836886        4.7323054252       14.4371947555
H                 6.0674618286        4.5632442529       15.9905526013
H                 7.6927243544        2.8233900156       15.3192611263
H                 6.1584482020        2.4044133571       15.1740212821
H                 7.8289769218        3.2112550153       13.0149070131
H                 6.2755779443        2.8605810454       12.8519549928
H                 8.2302004888        0.9509324357       13.7269364931
H                 7.6510696251        1.0040769018       12.2373424814
H                 6.6893628237        0.6152670733       13.4559401985
Si                3.9667810312       17.0616639361       14.4924567450
Si                5.6048658579       15.9520966303        2.8622876610
C                 4.6650112272       16.5003829797       10.1184348522
C                 4.5554688207       17.6570881175        9.3484865562
C                 4.2705213821       19.9491946759        8.3902578281
C                 4.4855007059       18.9819788478        7.4589516077
C                 4.6127414977       17.6613729956        7.9315124666
C                 4.8114567726       16.4516145464        7.2228483000
C                 4.9171463225       15.2982962713        8.0023611711
C                 5.2369470825       13.0078681197        8.9853869458
C                 5.0188654029       13.9833108388        9.9048260082
C                 4.8691610124       15.2980837608        9.4157928255
C                 4.1429910818       21.3759717104        8.1723980618
C                 3.8903474555       23.6631666979        7.2053132457
C                 3.7601015264       24.8389147705        6.4791140248
C                 3.5381224484       26.0207223005        7.1840586343
C                 3.3600506860       27.3206023994        6.6278963042
C                 3.1470319944       28.2606914873        7.5878962499
C                 3.4958018745       26.0123650548        8.6045751592
C                 3.6210248762       24.8271118501        9.3290031625
C                 3.8029561343       23.6453043200        8.6240585529
C                 3.9597895101       22.3185598675        9.1419625827
C                 5.3866840242       11.5827694921        9.2280437586
C                 5.6553481556        9.3237983603       10.2748333673
C                 5.7761133245        8.1833676685       11.0506266814
C                 5.9965276867        6.9791135266       10.3775639888
C                 6.1524118813        5.7073424569       10.9780067593
C                 6.3765889395        4.7106554128       10.0587448186
C                 6.1729708978        6.9467382732        8.9623611168
C                 6.0052677115        8.1064972392        8.2060511965
C                 5.7613748038        9.3055218001        8.8543168425
C                 5.6100988998       10.6152612662        8.2969146600
C                 4.5238058763       16.6452774511       11.5309808977
C                 4.3492776258       16.8002663497       12.7107889491
C                 3.1112588002       15.5287463986       15.1306264506
C                 3.9665925201       14.2531938464       15.0163829147
C                 3.3341747123       13.0091846420       15.6526042809
C                 2.1494957292       12.4227888916       14.8941688994
C                 2.5309693773       11.6833659848       13.6268626980
C                 1.3346593446       11.0975647353       12.9011948424
C                 5.6059368407       17.3211166576       15.3399843724
C                 5.5891824737       17.3373251952       16.8674824410
C                 6.9168728327       17.7673041409       17.4900654316
C                 8.1274866286       17.0296062145       16.9411650937
C                 9.4320420159       17.3916252114       17.6105082292
C                10.6442418666       16.7989936679       16.9578145392
C                 2.9236218198       18.6159977989       14.5836744521
C                 3.7256061419       19.8838059579       14.2177408934
C                 2.9320956243       21.1293715457       13.9384198760
C                 3.8029801044       22.3765625135       13.7524420267
C                 3.0871032570       23.6265909543       13.3764125277
C                 3.9811970281       24.8182077866       13.2676597663
C                 4.9217678506       16.3149169296        5.8028631405
C                 5.0659675921       16.1828092565        4.6129591487
C                 6.0534975722       14.1526982128        2.6922507703
C                 7.1255832240       13.6535220419        3.6683687962
C                 7.3944639093       12.1467164865        3.5950403871
C                 6.1939478469       11.2852368618        3.9290920287
C                 6.4972709525        9.7875512469        3.9687673032
C                 5.2783944510        8.9390343152        4.2346270668
C                 4.2425387343       16.4188503067        1.6642360683
C                 3.0503489350       15.4633609916        1.6144648534
C                 1.8858725865       15.9804581764        0.7741992182
C                 0.7042794616       15.0151686846        0.7138006976
C                -0.5200552405       15.6076266647        0.0612841294
C                -1.6730348576       14.6090179861       -0.0375498134
C                 7.0449743447       17.1117838646        2.6097120295
C                 6.6559818406       18.5925430187        2.7985238270
C                 7.8198578062       19.5445139185        2.6392913637
C                 7.3911263127       21.0243534529        2.8977120133
C                 7.1598379804       21.3753608860        4.3412543671
C                 6.8038833313       22.8569981157        4.5325458691
S                 4.2968700079       19.2507611976        9.9905529406
S                 5.1573252035       13.6829124390        7.3726933100
S                 4.1432697599       22.0619981623        6.5761767499
S                 3.2203059752       27.6132597590        9.1903168235
S                 5.3689303448       10.9441273545       10.8497706042
S                 6.3636856466        5.3134306918        8.4370179731
H                 4.5335006963       19.1832989469        6.5317191878
H                 5.0193846225       13.7999047329       10.8370178374
H                 3.8226791472       24.8389504409        5.5315126023
H                 3.3889452968       27.5080199178        5.6980070579
H                 2.9737348058       29.1754810748        7.3983759654
H                 3.5851658447       24.8294789636       10.2783758025
H                 3.9390860845       22.1151717835       10.0693721243
H                 5.7104463947        8.2188437393       11.9982281038
H                 6.1077767199        5.5579454669       11.9149808760
H                 6.5210165803        3.7983834356       10.2854606730
H                 6.0581346362        8.0721695586        7.2566785564
H                 5.6601519280       10.7915650239        7.3647228308
H                 2.2704200612       15.3966402507       14.6231726048
H                 2.8708097360       15.6695677837       16.0808846995
H                 4.8427465651       14.4187420112       15.4467887944
H                 4.1346621011       14.0690666907       14.0581541866
H                 3.0362983490       13.2425874456       16.5679695428
H                 4.0310399448       12.3103691985       15.7354972652
H                 1.5271344494       13.1559642072       14.6585969570
H                 1.6652442206       11.8000779305       15.4928404523
H                 3.0027201190       12.3076935280       13.0202206659
H                 3.1579544623       10.9524344481       13.8562353788
H                 0.7314414653       11.8199195828       12.6305527913
H                 1.6406775396       10.6115311855       12.1062723781
H                 0.8601081078       10.4812768063       13.4984494212
H                 5.9829179511       18.1817146650       15.0270102203
H                 6.2226524474       16.6047901720       15.0464936141
H                 4.8786306753       17.9546507721       17.1737259661
H                 5.3611307452       16.4316210877       17.1949805775
H                 6.8724091477       17.6212197986       18.4684860405
H                 7.0405279776       18.7374195944       17.3384492040
H                 8.2040116622       17.2222347853       15.9728404252
H                 7.9788582676       16.0563009057       17.0426558628
H                 9.3988608819       17.0902397336       18.5535044859
H                 9.5225922391       18.3771608325       17.6147591515
H                10.7387573720       17.1634548751       16.0525452177
H                11.4399869281       17.0221406065       17.4836890482
H                10.5467393660       15.8246000920       16.9098145419
H                 2.5649528057       18.7120231657       15.5016965404
H                 2.1559761909       18.5280605230       13.9642796531
H                 4.2728091470       19.6819272928       13.4169734110
H                 4.3508135598       20.0772907990       14.9597039511
H                 2.3932116021       20.9899255315       13.1194088522
H                 2.3050396097       21.2846105919       14.6887076565
H                 4.4763744919       22.1829963865       13.0538738006
H                 4.2917789267       22.5410725644       14.5966043406
H                 2.6347305222       23.4866396661       12.5083387761
H                 2.3893083152       23.8116068867       14.0528405337
H                 4.6598513958       24.6552056046       12.5791874805
H                 3.4493118380       25.6039166315       13.0219918835
H                 4.4201586140       24.9767276543       14.1290028911
H                 6.3719954263       13.9943373752        1.7676751768
H                 5.2348941620       13.6111193791        2.8215496559
H                 6.8451105279       13.8807959758        4.5909960503
H                 7.9707450617       14.1351116703        3.4839850428
H                 8.1294304004       11.9244083483        4.2208115694
H                 7.7019102595       11.9227039759        2.6816234646
H                 5.8388892551       11.5615258468        4.8106270342
H                 5.4868879093       11.4494170668        3.2554979709
H                 7.1681221377        9.6151291264        4.6742432781
H                 6.8929467404        9.5169070329        3.1031732563
H                 4.6313428924        9.0603866721        3.5087820894
H                 5.5408895243        7.9963062238        4.2845754035
H                 4.8711012213        9.2095795968        5.0833945465
H                 4.6325122129       16.4789228900        0.7563099203
H                 3.9104804022       17.3209924381        1.9058301505
H                 2.7314175353       15.3023352361        2.5381548380
H                 3.3506415327       14.5952721923        1.2433947636
H                 1.5768016426       16.8425117444        1.1512914478
H                 2.2071299463       16.1519501776       -0.1470110618
H                 0.4735332800       14.7338826834        1.6348338559
H                 0.9767924782       14.2065338304        0.2107748958
H                -0.8177780409       16.3932265823        0.5845018120
H                -0.2838872477       15.9195605313       -0.8484132362
H                -1.8907470605       14.2714794534        0.8554981066
H                -2.4588844325       15.0545579664       -0.4180073564
H                -1.4093185487       13.8609925133       -0.6146125114
H                 7.4045715766       16.9859719188        1.6968264723
H                 7.7622440825       16.8814590774        3.2519555357
H                 6.2683046449       18.7098188993        3.7018448091
H                 5.9577265919       18.8271760402        2.1378596577
H                 8.5362079403       19.2955331023        3.2749813646
H                 8.1848298002       19.4645942746        1.7216235189
H                 6.5595672744       21.2044485119        2.3929149938
H                 8.0938434268       21.6234251704        2.5381548380
H                 6.4233147070       20.8165835122        4.6972691071
H                 7.9767136845       21.1672574821        4.8602211274
H                 6.0220911401       23.0769060918        3.9852396270
H                 6.6012220038       23.0237616256        5.4748336387
H                 7.5629288051       23.4125714542        4.2562359217



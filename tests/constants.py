from datetime import datetime

emr_heath_expected_diagnosis_dates = [datetime(2023, 6, 20).date(), datetime(2023, 6, 20).date(), datetime(2023, 2, 21).date(), datetime(2023, 2, 21).date(), datetime(2019, 6, 3).date()]

emr_health_expected_output = '''aniridie diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-07 -- Hashed (anonymised) PID: 0x2bcaeb81
aphaque - pas de capsule presente diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-07 -- Hashed (anonymised) PID: 0x2bcaeb81
cataracte diagnosed by Dr Lazaros Konstantinidis on 2017-08-09 -- Hashed (anonymised) PID: 0x9ad4db34
décollement de la rétine diagnosed by Dr Lazaros Konstantinidis on 2017-08-08 -- Hashed (anonymised) PID: 0x61addad5
mydriase traumatique diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-07 -- Hashed (anonymised) PID: 0x6ada0972
trou maculaire (TM) diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0x3f1af163
cristallin clair diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0x5bc7a8d8
rétinopathie diabétique - background diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0x5bc7a8d8
pseudophaque diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0x9e64e19f
3+ cataracte corticale diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0x7bbe6ce3
membrane : épiretinienne diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0xb46c2546
opacification de la capsule postérieure diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0xb46c2546
pseudophake - chambre postérieure IOL diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0xb46c2546
cataracte diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0xbc36779a
membrane épirétinienne diagnosed by Dr Lazaros Konstantinidis on 2017-08-07 -- Hashed (anonymised) PID: 0xbc36779a
membrane épirétinienne diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-07 -- Hashed (anonymised) PID: 0xe9ef71
huile de silicone dans la cavité vitréenne diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-08 -- Hashed (anonymised) PID: 0x51b30603
mélanome de la choroïde diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-08 -- Hashed (anonymised) PID: 0x51b30603
membrane épirétinienne diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-08 -- Hashed (anonymised) PID: 0xf8020897
membrane épirétinienne diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-07 -- Hashed (anonymised) PID: 0x22530a25
membrane épirétinienne diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-08 -- Hashed (anonymised) PID: 0x6df932a1
stade III trou maculaire diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-08 -- Hashed (anonymised) PID: 0x900a12c0
décollement de la rétine rhegmatogène diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-09 -- Hashed (anonymised) PID: 0x1a57998b
ablation de l'huile de silicone diagnosed by Dre Alejandra Daruich-Matet on 2017-08-09 -- Hashed (anonymised) PID: 0x691d576e
œdème maculaire diabétique diagnosed by Dr Lazaros Konstantinidis on 2017-08-09 -- Hashed (anonymised) PID: 0xd39de390
décollement de la rétine diagnosed by Dr Lazaros Konstantinidis on 2017-08-10 -- Hashed (anonymised) PID: 0xac7eddce
pseudophaque - IOL subluxée diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-14 -- Hashed (anonymised) PID: 0x487791c2
membrane épirétinienne diagnosed by Prof. Thomas J. Wolfensberger on 2017-08-14 -- Hashed (anonymised) PID: 0xffaec439
rétinopathie diabétique proliférante diagnosed by  Migration De Données on 2017-08-11 -- Hashed (anonymised) PID: 0xab49125d
DMLA diagnosed by  Migration De Données on 2017-08-04 -- Hashed (anonymised) PID: 0xbae6452c
autre pathologie oculaire diagnosed by  Migration De Données on 2017-08-04 -- Hashed (anonymised) PID: 0x577fd592
DMLA diagnosed by  Migration De Données on 2017-08-03 -- Hashed (anonymised) PID: 0xe65c7587
autre pathologie oculaire diagnosed by  Migration De Données on 2017-08-08 -- Hashed (anonymised) PID: 0x70bcadaf
autre pathologie oculaire diagnosed by  Migration De Données on 2017-08-11 -- Hashed (anonymised) PID: 0xf71a32db
--------------------------------------------------------------------------------
Bilan:
                         docteur  nombre de patients
1      Dr Lazaros Konstantinidis                  14
2  Prof. Thomas J. Wolfensberger                  13
3           Migration De Données                   6
4    Dre Alejandra Daruich-Matet                   1
'''
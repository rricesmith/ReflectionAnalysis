import A00_SlurmUtil





layer_depths = [ [300, 1.7], [500, 1.7], [800, 1.7] ]

#f_values = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
f_values = [0.3]
#f_values = [0.1]
#dB_values = [40, 45, 50]
dB_values = [40]

#direction = 'above'
direction = 'below'

#MB Values
#f_values = [0.15, 0.1, 0.05, 0.01, 0.005]
#dB_values = [0]


if direction == 'above':
    dB_values = [0]
    layer_depths = [ [300, 1.7 ]]

def testLArray(f_values, dB_values):
    
    for f in f_values:
        for dB in dB_values:
            L = f * 10 ** -(dB/20)
            print(f'L {L} from {f}f {dB}dB')

testLArray(f_values, dB_values)
#quit() 


#cmd = 'python CoreAnalysis/C03_LPDA_coreObjectPklCondense.py'
cmd = 'python CoreAnalysis/C03_Gen2_coreObjectPklCondense.py'

for f in f_values:
    for dB in dB_values:

        args = f' {dB} {f} --direction {direction}'
        A00_SlurmUtil.makeAndRunJob(cmd + args, f'{f}f_{dB}dB_Condense', runDirectory='run/CoreRefl1.70km', partition='standard')

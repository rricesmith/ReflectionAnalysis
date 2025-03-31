import os

# This function removes templates that were deemed too abnormal or difficult to identify
# It keeps them elsewhere for reference
# It should be noted this essentially does reduce what the simulated rate would be
# But signals removed ideally represent events that aren't as likely
# In my personal eye-check, I think maybe 50% of the rate is removed at most

folder = "DeepLearning/templates/RCR/3.29.25"

bad_100s = [1586, 1587, 2057, 2832, 2833, 2835, 2837, 2841, 2880, 2949, 2951, 2952, 2954, 2969, 2970, 2972, 2973, 
           2974, 3027, 3028, 3029, 3064, 3065, 3080, 3081, 3299, 3383, 3409, 3411, 3412, 3413, 3894, 3895, 3896,
           3898, 3899]

bad_200s = [1096, 1097, 1098, 1125, 1471, 1473, 1474, 1476, 1596, 1598, 1599, 1600, 2299, 2300, 2302, 3017, 3019,
           3020, 3022, 3024, 3027, 3187, 3189, 3190, 3191, 3896, 3898, 3899, 3901, 3902, 3904, 4405, 4406, 4407]

for n in bad_100s:
    command = f"cp {folder}/100s_{n}* {folder}/bad/"
    print(command)
    os.system(command)
    command = f"rm {folder}/100s_{n}*"
    print(command)
    os.system(command)
for n in bad_200s:
    command = f"cp {folder}/200s_{n}* {folder}/bad/"
    print(command)
    os.system(command)
    command = f"rm {folder}/200s_{n}*"
    print(command)
    os.system(command)
print("Done")

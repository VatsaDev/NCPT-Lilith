# nanoChatGpt

a barebones Nanogpt, but finetuned on conversational data, finetuned with the lilith optimizer instead
The losses are wild, mostly due to bad datasets and terrible shuffles, but that'll be fixed in future tests

From the old work with AdamW:

"""
step 0: train loss 2.6743, val loss 2.4028
iter 0: loss 2.8581, time 113655.37ms, mfu -100.00%
iter 1: loss 2.4210, time 20114.44ms, mfu -100.00%
iter 2: loss 2.7040, time 20462.91ms, mfu -100.00%
iter 3: loss 2.7455, time 21079.82ms, mfu -100.00%
iter 4: loss 2.8193, time 21749.35ms, mfu -100.00%
step 5: train loss 2.6395, val loss 2.2629
iter 5: loss 2.3520, time 68664.59ms, mfu 1.48%
iter 6: loss 2.8830, time 21596.65ms, mfu 1.81%
iter 7: loss 2.4695, time 21596.16ms, mfu 2.10%
iter 8: loss 2.8103, time 20961.95ms, mfu 2.37%
iter 9: loss 2.6131, time 20878.13ms, mfu 2.62%
step 10: train loss 2.6102, val loss 2.2391
iter 10: loss 2.3422, time 61120.89ms, mfu 2.53%
iter 11: loss 2.3073, time 21920.15ms, mfu 2.74%
iter 12: loss 2.7338, time 21078.47ms, mfu 2.95%
iter 13: loss 2.3733, time 20695.33ms, mfu 3.15%
iter 14: loss 2.4036, time 20986.12ms, mfu 3.32%
step 15: train loss 2.5686, val loss 2.2425
iter 15: loss 2.3052, time 37848.45ms, mfu 3.25%
iter 16: loss 2.9466, time 20955.49ms, mfu 3.42%
iter 17: loss 2.2908, time 21006.01ms, mfu 3.56%
iter 18: loss 2.7633, time 21098.01ms, mfu 3.69%
iter 19: loss 2.6824, time 21146.96ms, mfu 3.80%
step 20: train loss 2.5153, val loss 2.1773
iter 20: loss 2.7183, time 68526.45ms, mfu 3.57%
iter 21: loss 2.5928, time 21832.74ms, mfu 3.68%
iter 22: loss 2.6953, time 21190.26ms, mfu 3.79%
iter 23: loss 2.6496, time 20822.73ms, mfu 3.90%
iter 24: loss 2.4620, time 20921.54ms, mfu 4.00%
step 25: train loss 2.4662, val loss 2.2525
iter 25: loss 2.0399, time 37521.94ms, mfu 3.87%
iter 26: loss 2.7660, time 21050.69ms, mfu 3.97%
iter 27: loss 2.6088, time 21028.87ms, mfu 4.05%
iter 28: loss 2.7657, time 21025.33ms, mfu 4.13%
iter 29: loss 2.2539, time 21018.04ms, mfu 4.20%
step 30: train loss 2.4787, val loss 2.2095
iter 30: loss 2.5411, time 37180.44ms, mfu 4.06%
iter 31: loss 2.4646, time 21099.89ms, mfu 4.13%
iter 32: loss 2.2167, time 21183.78ms, mfu 4.20%
iter 33: loss 2.5584, time 21077.22ms, mfu 4.27%
iter 34: loss 2.5078, time 20982.60ms, mfu 4.32%
step 35: train loss 2.4710, val loss 2.2371
iter 35: loss 2.0193, time 37211.46ms, mfu 4.17%
iter 36: loss 2.5665, time 21097.19ms, mfu 4.23%
iter 37: loss 2.3142, time 21080.44ms, mfu 4.29%
iter 38: loss 2.3900, time 21038.20ms, mfu 4.35%
iter 39: loss 2.2882, time 21029.18ms, mfu 4.40%
step 40: train loss 2.4815, val loss 2.1641
iter 40: loss 1.8365, time 64654.63ms, mfu 4.11%
iter 41: loss 2.5460, time 21894.29ms, mfu 4.17%
iter 42: loss 2.4421, time 21208.04ms, mfu 4.23%
iter 43: loss 2.1610, time 20804.73ms, mfu 4.30%
iter 44: loss 2.3204, time 20946.67ms, mfu 4.35%
step 45: train loss 2.5098, val loss 2.2341
iter 45: loss 2.5102, time 37608.92ms, mfu 4.19%
iter 46: loss 2.7543, time 21001.14ms, mfu 4.26%
iter 47: loss 2.3898, time 21018.38ms, mfu 4.31%
iter 48: loss 1.8896, time 21149.41ms, mfu 4.37%
iter 49: loss 2.6746, time 21073.53ms, mfu 4.41%
step 50: train loss 2.4871, val loss 2.2134
iter 50: loss 2.6299, time 37158.03ms, mfu 4.24%
"""

from lilith, scaling the grad_accum for the large batch_size lilith likes:

"""
step 0: train loss 2.9857, val loss 3.0947
iter 0: loss 1.7004, time 69641.57ms, lr:0.0003 , mfu: -100.00%
iter 1: loss 3.9049, time 14991.77ms, lr:0.0003 , mfu: -100.00%
iter 2: loss 4.7442, time 15328.89ms, lr:0.0003 , mfu: -100.00%
iter 3: loss 3.7679, time 15651.19ms, lr:0.0003 , mfu: -100.00%
iter 4: loss 2.8901, time 15997.77ms, lr:0.0003 , mfu: -100.00%
step 5: train loss 3.1744, val loss 3.7768
iter 5: loss 3.3399, time 18371.47ms, lr:0.0003 , mfu: 3.06%
iter 6: loss 5.7953, time 15697.27ms, lr:0.0003 , mfu: 3.11%
iter 7: loss 2.5299, time 15623.83ms, lr:0.0003 , mfu: 3.16%
iter 8: loss 2.6336, time 15642.95ms, lr:0.0003 , mfu: 3.20%
iter 9: loss 4.9486, time 15743.15ms, lr:0.0003 , mfu: 3.24%
step 10: train loss 3.0292, val loss 3.6096
iter 10: loss 3.1013, time 18190.61ms, lr:0.0003 , mfu: 3.22%
iter 11: loss 2.8059, time 15759.72ms, lr:0.0003 , mfu: 3.26%
iter 12: loss 2.6237, time 15730.44ms, lr:0.0003 , mfu: 3.29%
iter 13: loss 2.9243, time 15707.64ms, lr:0.0003 , mfu: 3.32%
iter 14: loss 2.2595, time 15704.12ms, lr:0.0003 , mfu: 3.34%
step 15: train loss 2.8511, val loss 3.2742
iter 15: loss 3.4352, time 18093.51ms, lr:0.0003 , mfu: 3.32%
iter 16: loss 4.6779, time 15715.52ms, lr:0.0003 , mfu: 3.34%
iter 17: loss 3.1418, time 15703.26ms, lr:0.0003 , mfu: 3.37%
iter 18: loss 2.6684, time 15704.44ms, lr:0.0003 , mfu: 3.39%
iter 19: loss 2.3027, time 15706.93ms, lr:0.0003 , mfu: 3.41%
step 20: train loss 2.6088, val loss 3.3652
iter 20: loss 2.4711, time 18098.05ms, lr:0.0003 , mfu: 3.38%
iter 21: loss 3.7580, time 15688.78ms, lr:0.0003 , mfu: 3.40%
iter 22: loss 2.7125, time 15704.24ms, lr:0.0003 , mfu: 3.41%
iter 23: loss 2.5005, time 15710.07ms, lr:0.0003 , mfu: 3.43%
iter 24: loss 2.6010, time 15740.14ms, lr:0.0003 , mfu: 3.44%
step 25: train loss 2.6621, val loss 3.0280
saving checkpoint to /content/drive/MyDrive/Model
iter 25: loss 3.0311, time 26169.83ms, lr:0.0003 , mfu: 3.31%
iter 26: loss 1.1536, time 15952.90ms, lr:0.0003 , mfu: 3.33%
iter 27: loss 1.7893, time 15841.74ms, lr:0.0003 , mfu: 3.35%
iter 28: loss 2.4909, time 15703.62ms, lr:0.0003 , mfu: 3.38%
iter 29: loss 2.3683, time 15648.66ms, lr:0.0003 , mfu: 3.40%
step 30: train loss 2.6979, val loss 3.1413
iter 30: loss 3.0729, time 18088.19ms, lr:0.0003 , mfu: 3.37%
iter 31: loss 2.6453, time 15730.13ms, lr:0.0003 , mfu: 3.39%
iter 32: loss 4.0868, time 15767.80ms, lr:0.0003 , mfu: 3.41%
iter 33: loss 2.2032, time 15759.81ms, lr:0.0003 , mfu: 3.42%
iter 34: loss 3.2012, time 15755.55ms, lr:0.0003 , mfu: 3.44%
step 35: train loss 2.7145, val loss 2.8049
saving checkpoint to /content/drive/MyDrive/Model
iter 35: loss 2.5128, time 26018.97ms, lr:0.0003 , mfu: 3.31%
iter 36: loss 2.1513, time 15914.08ms, lr:0.0003 , mfu: 3.33%
iter 37: loss 2.4486, time 15827.47ms, lr:0.0003 , mfu: 3.35%
iter 38: loss 2.1967, time 15707.33ms, lr:0.0003 , mfu: 3.37%
iter 39: loss 2.1392, time 15661.34ms, lr:0.0003 , mfu: 3.39%
step 40: train loss 2.5543, val loss 2.6940
saving checkpoint to /content/drive/MyDrive/Model
iter 40: loss 2.6238, time 26106.85ms, lr:0.0003 , mfu: 3.27%
iter 41: loss 2.9456, time 15975.14ms, lr:0.0003 , mfu: 3.29%
iter 42: loss 2.0905, time 15873.15ms, lr:0.0003 , mfu: 3.32%
iter 43: loss 2.8748, time 15697.87ms, lr:0.0003 , mfu: 3.34%
iter 44: loss 3.3145, time 15632.69ms, lr:0.0003 , mfu: 3.37%
step 45: train loss 2.4751, val loss 2.8284
iter 45: loss 3.1328, time 18048.55ms, lr:0.0003 , mfu: 3.34%
iter 46: loss 2.8454, time 15742.19ms, lr:0.0003 , mfu: 3.37%
iter 47: loss 3.8444, time 15773.51ms, lr:0.0003 , mfu: 3.38%
iter 48: loss 2.0577, time 15741.99ms, lr:0.0003 , mfu: 3.40%
iter 49: loss 2.5562, time 15710.72ms, lr:0.0003 , mfu: 3.42%
step 50: train loss 2.4695, val loss 2.6476
saving checkpoint to /content/drive/MyDrive/Model

"""

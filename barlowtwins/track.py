import matplotlib.pyplot as plt
import pandas as pd
import time


while True:
    with open('checkpoint/stats_1.txt') as f:
        stats = f.readlines()

    df = pd.DataFrame([eval(s.replace('\n','')) for s in stats[1:] if 'NaN' not in s])

    plt.plot(df['step'], df['lr_weights'])
    plt.savefig('lr_curve')
    
    plt.cla()
    plt.clf()
    plt.close("all")
        
    plt.plot(df['step'], df['loss'])
    plt.savefig('loss_curve')
    
    plt.cla()
    plt.clf()
    plt.close("all")
    
    print("plots saved")
    time.sleep(3600//2)
    
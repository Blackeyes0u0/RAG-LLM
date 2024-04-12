import pandas as pd

train = pd.read_csv('./open/train.csv')
test = pd.read_csv('./open/test.csv')

if __name__=='__main__':
    print('start')
    for step,i in enumerate(range(len(train))):
        if step>111:
            x = list(train.loc[i])
            for j in x:
                print(j)
                print()
            breakpoint()
        
        
        

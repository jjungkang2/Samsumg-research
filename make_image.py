import matplotlib.pyplot as plt
import numpy as np
import argparse

def make_np(f):
    result = []
    for line in f.readlines():
        result.append(list(map(float, line.split())))
    result = np.array(result)

    return result

def compare_three():
    f1 = open("result/BGD.txt", 'r')
    f2 = open("result/miniBGD.txt", 'r')
    f3 = open("result/SGD.txt", 'r')

    result1 = make_np(f1)
    result2 = make_np(f2)
    result3 = make_np(f3)

    plt.clf()
    plt.plot(result1[:,1], 'r', label='Gradient Descent')
    plt.plot(result2[:,1], 'g', label='Mini-Batch Gradient Descent') 
    plt.plot(result3[:,1], 'b', label='Stochastic Gradient Descent')
    plt.xlim(0,200)
    plt.ylim(0,4)
    plt.legend()

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.title('Gradient Descent Compare Graph')

    plt.show()

def show_each(result, col, name):
    plt.clf()
    plt.plot(result[:,1], col, label=name)
    plt.xlim(0,200)
    plt.ylim(0,10)
    plt.legend()

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)

    plt.show()
    plt.show()

def compare_five():
    f1 = open("result/momentum.txt", 'r')
    f2 = open("result/NAG.txt", 'r')
    f3 = open("result/Adagrad.txt", 'r')
    f4 = open("result/RMSProp.txt", 'r')
    f5 = open("result/Adam.txt", 'r')
    
    result1 = make_np(f1)
    result2 = make_np(f2)
    result3 = make_np(f3)
    result4 = make_np(f4)
    result5 = make_np(f5)

    show_each(result1, 'r', 'momentum')
    show_each(result2, 'g', 'NAG')
    show_each(result3, 'b', 'Adagrad')
    show_each(result4, 'c', 'RMSProp')
    show_each(result5, 'm', 'Adam')
    

    plt.clf()
    plt.plot(result1[:,1], 'r', label='momentum')
    plt.plot(result2[:,1], 'g', label='NAG') 
    plt.plot(result3[:,1], 'b', label='Adagrad')
    plt.plot(result4[:,1], 'c', label='RMSProp')
    plt.plot(result5[:,1], 'm', label='Adam')
    plt.xlim(0,200)
    # plt.ylim(0,1)
    plt.legend()

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)

    plt.show()
    plt.show()


def compare_five_big():
    f1 = open("result/big_momentum.txt", 'r')
    f2 = open("result/big_NAG.txt", 'r')
    f3 = open("result/big_Adagrad.txt", 'r')
    f4 = open("result/big_RMSProp.txt", 'r')
    f5 = open("result/big_Adam.txt", 'r')

    
    result1 = make_np(f1)
    result2 = make_np(f2)
    result3 = make_np(f3)
    result4 = make_np(f4)
    result5 = make_np(f5)

    show_each(result1, 'r', 'momentum')
    show_each(result2, 'g', 'NAG')
    show_each(result3, 'b', 'Adagrad')
    show_each(result4, 'c', 'RMSProp')
    show_each(result5, 'm', 'Adam')
    

    plt.clf()
    plt.plot(result1[:,1], 'r', label='momentum')
    plt.plot(result2[:,1], 'g', label='NAG') 
    plt.plot(result3[:,1], 'b', label='Adagrad')
    plt.plot(result4[:,1], 'c', label='RMSProp')
    plt.plot(result5[:,1], 'm', label='Adam')
    plt.xlim(0,200)
    plt.ylim(0,1)
    plt.legend()

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)

    plt.show()

def main(args):
    if args.mode==1:
        compare_three()
    elif args.mode==2:
        compare_five()
    elif args.mode==3:
        compare_five_big()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, default=2)
    args = parser.parse_args()
    
    main(args)
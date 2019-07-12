import func as f
from BP import BP

if __name__ == "__main__":
    x1, y1 = f.generate_linear(n = 100)
    x2, y2 = f.generate_XOR_easy()
    inp = int(input())
    if (inp == 1 ):
        network1 = BP(x1, y1)
        network1.train()
        network1.test()
        f.show_result(network1.x, network1.y, network1.pred_y)
    else:  
        network2 = BP(x2, y2)
        network2.train()
        network2.test()
        f.show_result(network2.x, network2.y, network2.pred_y)



